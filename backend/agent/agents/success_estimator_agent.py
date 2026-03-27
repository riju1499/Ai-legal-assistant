import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SuccessEstimatorAgent:
    """
    Estimate success_probability using the LLM based on facts, retrieved laws/cases,
    and the current strategy draft. Returns:
      - success_probability: { point: float (0-1), ci: [low, high] }
      - reasoning: str (brief justification)
    """

    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        import json, re
        if not text:
            return None
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        candidate = m.group(0)
        # Light repairs
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # remove trailing commas
        if "'" in candidate and '"' not in candidate:
            candidate = candidate.replace("'", '"')
        try:
            return json.loads(candidate)
        except Exception:
            # Numeric fallback
            nums = [float(x) for x in re.findall(r"0?\.\d+|1\.0|0\.0", candidate)]
            if len(nums) >= 3:
                point, low, high = nums[0], nums[1], nums[2]
                return {
                    "success_probability": {"point": point, "ci": [low, high]},
                    "reasoning": "Parsed from numeric fallback."
                }
            return None

    def run(
        self,
        facts: str,
        laws: Optional[List[Dict[str, Any]]],
        cases: Optional[List[Dict[str, Any]]],
        draft_strategy: Optional[Dict[str, Any]],
        case_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        import json

        laws = (laws or [])[:10]
        cases = (cases or [])[:10]
        draft = draft_strategy or {}

        desired_outcome = draft.get("desired_outcome") or draft.get("goal") or ""
        documents = draft.get("documents") or draft.get("required_documents") or []
        precedents = draft.get("precedents") or draft.get("mined_precedents") or []
        strengths = draft.get("strengths") or []
        weaknesses = draft.get("weaknesses") or []

        def summarize_precedents(precs):
            out = []
            for p in precs[:8]:
                title = p.get("title") or p.get("case_number_english") or "Case"
                principle = p.get("principle") or p.get("winning_point") or p.get("summary") or ""
                out.append({"title": str(title)[:120], "principle": str(principle)[:200]})
            return out

        prompt = (
            "You are a Nepali legal outcomes analyst. Estimate the probability of success for the party seeking the desired outcome.\n"
            "Return STRICT JSON only (no code fences, no commentary).\n\n"
            "Constraints:\n"
            "- success_probability.point must be within [0.35, 0.85].\n"
            "- success_probability.ci is a 95% interval [low, high], 0.0–1.0, and must contain point.\n"
            "- reasoning: 2–4 short lines, case-specific.\n\n"
            f"Case Type: {case_type or 'Unknown'}\n"
            f"Desired Outcome: {desired_outcome}\n"
            f"Facts: {facts}\n\n"
            "Applicable Laws (truncated):\n"
            f"{json.dumps(laws, ensure_ascii=False)[:2000]}\n\n"
            "Similar Cases (truncated):\n"
            f"{json.dumps(cases, ensure_ascii=False)[:2000]}\n\n"
            "Precedents (summary):\n"
            f"{json.dumps(summarize_precedents(precedents), ensure_ascii=False)}\n\n"
            "Documents (if any):\n"
            f"{json.dumps(documents, ensure_ascii=False)[:1000]}\n\n"
            "Strategy signals (if any):\n"
            f"Strengths: {json.dumps([str(s) for s in strengths][:8], ensure_ascii=False)}\n"
            f"Weaknesses: {json.dumps([str(w) for w in weaknesses][:8], ensure_ascii=False)}\n\n"
            "Output JSON schema exactly:\n"
            "{\n"
            '  "success_probability": { "point": 0.55, "ci": [0.4, 0.7] },\n'
            '  "reasoning": "..." \n'
            "}"
        )

        for temperature in (0.1, 0.3):
            try:
                text = self.llm.generate(prompt, max_tokens=600, temperature=temperature)
                data = self._safe_parse_json(text)
                if data:
                    sp = (data or {}).get("success_probability") or {}
                    point = sp.get("point")
                    ci = sp.get("ci") or []

                    def clamp(x: float, lo: float, hi: float) -> float:
                        return max(lo, min(hi, float(x)))

                    if isinstance(point, (int, float)):
                        point = clamp(point, 0.35, 0.85)
                    else:
                        point = 0.55

                    if isinstance(ci, list) and len(ci) == 2 and all(isinstance(v, (int, float)) for v in ci):
                        low = clamp(ci[0], 0.0, 1.0)
                        high = clamp(ci[1], 0.0, 1.0)
                        if low > high:
                            low, high = high, low
                        if not (low <= point <= high):
                            low = clamp(point - 0.15, 0.0, 1.0)
                            high = clamp(point + 0.15, 0.0, 1.0)
                        ci = [round(low, 2), round(high, 2)]
                    else:
                        ci = [round(max(0.0, point - 0.15), 2), round(min(1.0, point + 0.15), 2)]

                    return {
                        "success_probability": {"point": round(point, 2), "ci": ci},
                        "reasoning": data.get("reasoning") or "Estimated from facts, laws, and precedents."
                    }
            except Exception as e:
                logger.warning(f"SuccessEstimatorAgent attempt failed: {e}")

        return {
            "success_probability": {"point": 0.55, "ci": [0.4, 0.7]},
            "reasoning": "Default fallback based on limited inputs."
        }


