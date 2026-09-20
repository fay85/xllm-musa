#!/usr/bin/env python3
"""Reference-free exact-arithmetic audit for GSM8K draft solutions.

The audit never reads the dataset answer.  It extracts explicit equations from
the model's selected draft and recomputes their left-hand side with Fraction.
It is deliberately conservative: prose-only reasoning is not judged here.
"""

import argparse
import ast
from fractions import Fraction
import json
import re
from pathlib import Path


NUMBER = r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d[\d,]*)?"
RHS_NUMBER_RE = re.compile(rf"^\s*({NUMBER})")
ALLOWED_EXPRESSION_RE = re.compile(r"[-+*/().\d\s]+")


def parse_number(value):
    return Fraction(value.replace("$", "").replace(",", "").replace(" ", ""))


def evaluate_node(node):
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported expression")


def normalize_lhs(raw):
    normalized = raw.replace("×", "*").replace("÷", "/").replace("−", "-")
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = re.sub(r"(?<=\d)-(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(
        r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized
    )
    # A slash followed by a unit is dimensional prose, not division.  Keep
    # numeric fractions and arithmetic division intact.
    normalized = re.sub(r"/\s*[A-Za-z]+", " ", normalized)
    normalized = re.sub(r"[A-Za-z_%]+", " ", normalized)
    normalized = re.sub(r"[^-+*/().\d\s]", " ", normalized)
    runs = ALLOWED_EXPRESSION_RE.findall(normalized)
    candidates = []
    for run in runs:
        run = run.strip()
        run = re.sub(r"[+*/.-]+$", "", run).strip()
        if not re.search(r"\d\s*[-+*/]", run):
            continue
        # Preserve a leading parenthesized negative number, for example
        # ``(-48) + 21 + (-3)``.  Starting at the first digit would discard
        # the opening parenthesis and force the parser onto a partial suffix.
        if re.match(r"(?:\(\s*-|-)\d", run):
            candidates.append(run)
        # Prose before the expression may leave unrelated numbers separated by
        # spaces.  Retain the final parseable suffix containing an operator.
        starts = [match.start() for match in re.finditer(r"(?<![\d.])[-+]?\d", run)]
        for start in starts:
            candidate = run[start:].strip()
            if re.search(r"\d\s*[-+*/]", candidate):
                candidates.append(candidate)
    for candidate in candidates:
        try:
            tree = ast.parse(candidate, mode="eval")
            return candidate, evaluate_node(tree)
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
    return None, None


def is_decimal_shift(left, right):
    """Return true when two nonzero values differ only by 1-3 decimal places."""
    if not left or not right:
        return False
    ratio = abs(left / right)
    return ratio in {Fraction(10), Fraction(100), Fraction(1000),
                     Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)}


def matches_prior_operand_transcription(lhs, claimed, prior_values):
    """Recognize a dropped/extra zero when the claimed result remains correct.

    Models occasionally write ``3 * 180 = 5400`` immediately after deriving
    ``1800``.  The printed operand is inconsistent, but changing the final
    answer would be wrong.  Only accept a one-token decimal shift to a value
    established in one of the previous three lines.
    """
    token_re = re.compile(r"\d+(?:\.\d+)?(?:\s*/\s*\d+)?")
    matches = list(token_re.finditer(lhs))
    for match in matches:
        try:
            operand = parse_number(match.group(0))
        except (ValueError, ZeroDivisionError):
            continue
        for prior_line, prior_value in prior_values:
            if not is_decimal_shift(operand, prior_value):
                continue
            replacement = (
                str(prior_value.numerator)
                if prior_value.denominator == 1
                else f"({prior_value.numerator}/{prior_value.denominator})"
            )
            rewritten = lhs[: match.start()] + replacement + lhs[match.end() :]
            try:
                computed = evaluate_node(ast.parse(rewritten, mode="eval"))
            except (SyntaxError, ValueError, ZeroDivisionError):
                continue
            if computed == claimed:
                return True
    return False


def matches_rate_conversion(line, claimed):
    """Ignore dimensional ``rate per interval * intervals`` bookkeeping.

    For example, ``2 per 15 minutes * 4 = 8 per hour`` intentionally uses
    the first number as a rate and the last number as a unit conversion.  The
    generic expression parser drops the prose and would incorrectly evaluate
    ``15 * 4``.  Keep this narrow exception tied to an explicit ``per`` rate,
    a multiplication sign, and an exact first-number-times-last-number result.
    """
    lhs = line.split("=", 1)[0]
    if not re.search(r"\bper\b", lhs, re.IGNORECASE):
        return False
    if not re.search(r"(?:×|\*)", lhs):
        return False
    numbers = []
    for token in re.findall(NUMBER, lhs):
        try:
            numbers.append(parse_number(token))
        except (ValueError, ZeroDivisionError):
            continue
    if len(numbers) < 3:
        return False
    return numbers[0] * numbers[-1] == claimed


def audit_content(content):
    discrepancies = []
    prior_values = []
    for line_number, line in enumerate(content.splitlines(), 1):
        prior_values = [
            (prior_line, value)
            for prior_line, value in prior_values
            if line_number - prior_line <= 3
        ]
        if "=" not in line:
            continue
        segments = line.split("=")
        checks = []
        for position in range(len(segments) - 1):
            raw_lhs = segments[position]
            if (
                r"\frac" in raw_lhs
                or re.search(r"\d[A-Za-z]|\b[A-Za-z]\b\s*[-+*/]", raw_lhs)
                or re.search(r"[-+×÷]\s*[A-Za-z]", raw_lhs)
            ):
                continue
            lhs, computed = normalize_lhs(raw_lhs)
            if lhs is None:
                continue
            match = RHS_NUMBER_RE.match(segments[position + 1])
            if not match:
                continue
            rhs_tail = segments[position + 1][match.end() :]
            # In chained equations such as ``a = 2*(b) = c``, the first RHS
            # is another expression rather than a claimed scalar result.
            if re.search(r"[-+*/×÷]\s*(?:\$?\s*\d|\()", rhs_tail):
                continue
            try:
                claimed = parse_number(match.group(1))
            except (ValueError, ZeroDivisionError):
                continue
            error = abs(float(computed - claimed))
            is_correct = computed == claimed or error <= max(
                1e-9, abs(float(computed)) * 1e-3
            )
            if is_correct:
                checks.append((lhs, None))
                prior_values.append((line_number, claimed))
                continue
            approximate = "≈" in line or re.search(
                r"\b(?:about|approximately|approx)\b", line, re.IGNORECASE
            )
            if approximate and error <= max(
                1e-9, abs(float(computed)) * 1e-3
            ):
                checks.append((lhs, None))
                continue
            nearby_correction = False
            for candidate in content.splitlines()[line_number : line_number + 2]:
                if "=" not in candidate:
                    continue
                correction_match = RHS_NUMBER_RE.match(candidate.split("=", 1)[1])
                if correction_match is None:
                    continue
                try:
                    corrected_claim = parse_number(correction_match.group(1))
                except (ValueError, ZeroDivisionError):
                    continue
                if corrected_claim == computed:
                    nearby_correction = True
                    break
            if nearby_correction:
                checks.append((lhs, None))
                prior_values.append((line_number, computed))
                continue
            if matches_rate_conversion(line, claimed):
                checks.append((lhs, None))
                prior_values.append((line_number, claimed))
                continue
            if re.search(
                r"\b(?:wait|seems? (?:off|wrong)|check (?:again|carefully)|"
                r"not (?:right|correct)|incorrect)\b",
                line,
                re.IGNORECASE,
            ):
                # The draft explicitly rejects this intermediate calculation;
                # audit the subsequent corrected line instead.
                checks.append((lhs, None))
                continue
            if matches_prior_operand_transcription(lhs, claimed, prior_values):
                checks.append((lhs, None))
                prior_values.append((line_number, claimed))
                continue
            checks.append(
                (
                    lhs,
                    {
                        "line_number": line_number,
                        "line": line.strip(),
                        "expression": lhs,
                        "computed": str(computed),
                        "claimed": str(claimed),
                    },
                )
            )
        # TeX inline equations often omit ``=`` and use ``\div``/``\times``.
        # Audit those separately without trying to interpret general prose.
        for latex in re.findall(r"\\\(([^()]+)\\\)", line):
            normalized_latex = latex.replace(r"\div", "/").replace(r"\times", "*")
            match = re.fullmatch(
                rf"\s*({NUMBER})\s*([+*/-])\s*({NUMBER})\s*=\s*({NUMBER})\s*",
                normalized_latex,
            )
            if match is None:
                continue
            expression = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            lhs, computed = normalize_lhs(expression)
            if lhs is None:
                continue
            claimed = parse_number(match.group(4))
            error = abs(float(computed - claimed))
            if computed == claimed or error <= max(
                1e-9, abs(float(computed)) * 1e-3
            ):
                continue
            checks.append(
                (
                    lhs,
                    {
                        "line_number": line_number,
                        "line": line.strip(),
                        "expression": lhs,
                        "computed": str(computed),
                        "claimed": str(claimed),
                    },
                )
            )
        # If a response corrects the same expression later on the same line,
        # only the final statement is actionable.
        final_checks = {}
        for lhs, discrepancy in checks:
            final_checks[re.sub(r"\s+", "", lhs)] = discrepancy
        discrepancies.extend(item for item in final_checks.values() if item is not None)
    return discrepancies


def selected_content(record):
    attempts = record.get("attempts", [])
    if not attempts:
        return record.get("content", "")
    selected_stage = record.get("selected_stage", "")
    if selected_stage.endswith("finalization"):
        for attempt in reversed(attempts):
            if attempt.get("stage", "").endswith("finalization"):
                return attempt.get("content", "")
    return attempts[0].get("content", "")


def self_test():
    cases = {
        "3 × 60 = 120 miles": (1, "180", "120"),
        "13 - 11 = 1 lego set": (1, "2", "1"),
        "Total = 360 + 160 = 500": (1, "520", "500"),
        "495 students × $15.00/student = $7,500.00": (1, "7425", "7500"),
        "2*(1.5 + 6) = 2*(7.5) = 15 miles": (0, None, None),
        r"\( 20 + 2 = 25 \) dollars": (1, "22", "25"),
        "20 - 5 = 19 video games": (1, "15", "19"),
        "5% of 350000 = 0.05 * 350000 = 17000? Wait, check:\n"
        "0.05 * 350000 = 17500.": (0, None, None),
        r"Number on Monday = \( 56 \div 4 = 12\) apples.": (1, "14", "12"),
        "4 - 2 - 0.5 = 1.5 hours": (0, None, None),
        "40,000 / 1.3 = 30,769.230769": (0, None, None),
        "Sum = (-48) + 21 + (-3) = -30": (0, None, None),
        "- 6 × 2 = 12, correct.": (0, None, None),
        "-6 × 2 = -11": (1, "-12", "-11"),
        r"\[ \frac{c}{2} + 2 = 5 \]": (0, None, None),
        "90 + second year + 45 = 200": (0, None, None),
        "2 per 15 minutes × 4 = 8 per hour.\n"
        "Hotel per day = 1500 + 300 = 1800 dollars.\n"
        "For 3 days, hotel cost = 3 * 180 = 5400 dollars.\n"
        "Total = 10000 + 5400 = 15400 dollars.": (0, None, None),
    }
    for text, expected in cases.items():
        found = audit_content(text)
        assert len(found) == expected[0], (text, found)
        if found:
            assert found[0]["computed"] == expected[1], (text, found)
            assert found[0]["claimed"] == expected[2], (text, found)
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input or not args.output_dir:
        parser.error("--input and --output-dir are required")

    records = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        discrepancies = audit_content(selected_content(record))
        records.append(
            {
                "dataset_index": record["dataset_index"],
                "prediction": record.get("prediction"),
                "has_arithmetic_discrepancy": bool(discrepancies),
                "discrepancies": discrepancies,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    flagged = [r for r in records if r["has_arithmetic_discrepancy"]]
    summary = {
        "input": str(Path(args.input).resolve()),
        "records": len(records),
        "flagged": len(flagged),
        "flagged_indices": [r["dataset_index"] for r in flagged],
        "selection_uses_reference": False,
        "scope": "explicit arithmetic equations only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
