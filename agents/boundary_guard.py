"""
BoundaryGuard (agents/boundary_guard.py)

Local projection of the SANGRE Boundary Guard Rules onto this codebase.

The canonical rules live in the Airtable base "NUMARA - Session State"
(appdAZR1Yivj0XKRX, table "Boundary Guard Rules"). Those rules govern crossings
between structure, meaning, memory, and behaviour in the symbolic kernel. This
module implements the subset that applies to *code synthesis in this repo*, using
the same schema and the same decision vocabulary, so the two stay legible to each
other.

Guards implemented here, and the canonical rule each projects from:

  guard_010_complexity_without_gain (audit_required)
      "Creating a new field, table, script, automation, relation type, or
      external layer without measurable behavioral gain."
      -> a synthesized tool must declare a measurable gain and a validation
         criterion before it may be written.

  guard_013_no_ingestion_without_rewind (stop_run)
      "Writing ... without a rollback manifest containing exact created/updated
      record IDs."
      -> a run that writes files must record a rollback manifest naming every
         file it created.

  guard_005_experiment_as_proof (audit_required)
      "Claiming exploratory probes prove a theory or treatment without a
      predefined control arm and frozen protocol."
      -> a run may not report a performance gain it did not measure.

This module deliberately does not write to Airtable. Canonical guard_009
("Audits should observe before modifying state") makes a preflight gate a poor
place to mutate the base; the Runtime Action Log is kept locally as JSONL and can
be synced upward as an explicit, separate step.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Decision vocabulary, matching the canonical "Response" single-select.
ALLOW = "allow"
REJECT = "reject"
STOP_RUN = "stop_run"
AUDIT_REQUIRED = "audit_required"

# Ordered by increasing severity, so the strictest triggered guard wins.
_DECISION_RANK = {ALLOW: 0, AUDIT_REQUIRED: 1, REJECT: 2, STOP_RUN: 3}

PROTOCOL_VERSION = "boundary-guard-local-v0.1"

# Phrases that assert a measured result. Used by guard_005.
_PROOF_CLAIM = re.compile(
    r"\b(\d+(\.\d+)?\s*x\s*(faster|speedup|throughput)"
    r"|proves?|proven|confirms?|confirmed|verified)\b",
    re.IGNORECASE,
)


@dataclass
class GuardRule:
    """A single boundary rule, mirroring the canonical Airtable field set."""

    guard_id: str
    layer: str
    forbidden_action: str
    detection_method: str
    severity: str
    response: str
    failure_class: str
    rationale: str
    version: str = "v0.1"


@dataclass
class PreflightDecision:
    """Result of a preflight check, mirroring the canonical Runtime Action Log."""

    action_id: str
    action_type: str
    decision: str
    triggered_guards: List[str] = field(default_factory=list)
    decision_rationale: str = ""
    write_target: Optional[str] = None
    executed: bool = False
    protocol_version: str = PROTOCOL_VERSION
    created_at: float = field(default_factory=time.time)

    @property
    def allowed(self) -> bool:
        """True only when nothing blocked the action."""
        return self.decision == ALLOW


CODE_SYNTHESIS_GUARDS: List[GuardRule] = [
    GuardRule(
        guard_id="guard_010_complexity_without_gain",
        layer="Cross-Layer",
        forbidden_action=(
            "Creating a new script or extension without measurable behavioral gain."
        ),
        detection_method=(
            "Proposed addition has no stated measurable gain or no associated "
            "validation/failure criterion."
        ),
        severity="high",
        response=AUDIT_REQUIRED,
        failure_class="F5_contract_violation",
        rationale="Implements BEAM constraint as a guard rule.",
    ),
    GuardRule(
        guard_id="guard_013_no_ingestion_without_rewind",
        layer="Cross-Layer",
        forbidden_action="Writing generated files without a rollback manifest.",
        detection_method=(
            "A synthesis run writes files and no rollback manifest naming every "
            "created file is produced in the same run."
        ),
        severity="critical",
        response=STOP_RUN,
        failure_class="F5_contract_violation",
        rationale="Provides the rewind button against accidental poisoning.",
    ),
    GuardRule(
        guard_id="guard_005_experiment_as_proof",
        layer="Cross-Layer",
        forbidden_action=(
            "Claiming a performance gain without a predefined control arm and "
            "frozen protocol."
        ),
        detection_method=(
            "Reported result asserts a speedup, proof, or confirmation that no "
            "measurement in the run produced."
        ),
        severity="high",
        response=AUDIT_REQUIRED,
        failure_class="F5_contract_violation",
        rationale="Prevents overclaiming from exploratory logs.",
    ),
]


class BoundaryGuard:
    """
    Preflight gate for actions that create or execute generated code.

    Callers describe an action, receive a PreflightDecision, and must not proceed
    unless `decision.allowed` is True. Every decision is appended to the local
    Runtime Action Log regardless of outcome.
    """

    def __init__(
        self,
        rules: Optional[List[GuardRule]] = None,
        log_path: str = "config/runtime_action_log.jsonl",
        manifest_path: str = "config/rollback_manifest.json",
    ):
        self.rules = rules if rules is not None else list(CODE_SYNTHESIS_GUARDS)
        self.log_path = log_path
        self.manifest_path = manifest_path

    def _rule(self, guard_id: str) -> GuardRule:
        for rule in self.rules:
            if rule.guard_id == guard_id:
                return rule
        raise KeyError(guard_id)

    def preflight_code_synthesis(
        self,
        tool_name: str,
        write_target: str,
        measurable_gain: Optional[str] = None,
        validation_criterion: Optional[str] = None,
        rollback_manifest: bool = True,
    ) -> PreflightDecision:
        """
        Check a proposed code-synthesis write against the active guards.

        Args:
            tool_name: Name of the tool being synthesized.
            write_target: Path the run intends to write.
            measurable_gain: What measurable behavioral gain this tool provides.
            validation_criterion: How to tell whether the tool works.
            rollback_manifest: Whether the run will record a rollback manifest.
        """
        decision = PreflightDecision(
            action_id=f"act_{int(time.time() * 1000)}",
            action_type="code_synthesis",
            decision=ALLOW,
            write_target=write_target,
        )
        reasons: List[str] = []

        if not (measurable_gain and measurable_gain.strip()) or not (
            validation_criterion and validation_criterion.strip()
        ):
            rule = self._rule("guard_010_complexity_without_gain")
            decision.triggered_guards.append(rule.guard_id)
            reasons.append(
                f"{rule.guard_id}: tool '{tool_name}' declares no measurable gain "
                f"and/or no validation criterion."
            )
            decision.decision = self._escalate(decision.decision, rule.response)

        if not rollback_manifest:
            rule = self._rule("guard_013_no_ingestion_without_rewind")
            decision.triggered_guards.append(rule.guard_id)
            reasons.append(f"{rule.guard_id}: run would write {write_target} with no rollback manifest.")
            decision.decision = self._escalate(decision.decision, rule.response)

        decision.decision_rationale = (
            " | ".join(reasons) if reasons else "No guard triggered; write permitted."
        )
        self._log(decision)
        return decision

    def preflight_result_claim(self, report: Dict[str, Any], measured_keys: List[str]) -> PreflightDecision:
        """
        Check a result report for performance claims the run never measured.

        Args:
            report: The report dictionary about to be returned to a caller.
            measured_keys: Keys in `report` that hold genuinely measured values.
        """
        decision = PreflightDecision(
            action_id=f"act_{int(time.time() * 1000)}",
            action_type="result_claim",
            decision=ALLOW,
        )
        reasons: List[str] = []

        for key, value in report.items():
            if key in measured_keys or not isinstance(value, str):
                continue
            if _PROOF_CLAIM.search(value):
                rule = self._rule("guard_005_experiment_as_proof")
                if rule.guard_id not in decision.triggered_guards:
                    decision.triggered_guards.append(rule.guard_id)
                reasons.append(f"{rule.guard_id}: field '{key}' claims {value!r} with no measurement behind it.")
                decision.decision = self._escalate(decision.decision, rule.response)

        decision.decision_rationale = (
            " | ".join(reasons) if reasons else "No unmeasured claims found."
        )
        self._log(decision)
        return decision

    def record_rollback_manifest(self, run_id: str, created_files: List[str]) -> Dict[str, Any]:
        """
        Record every file a run created, so the run can be rewound.

        Satisfies guard_013. Returns the manifest entry that was stored.
        """
        entry = {
            "run_id": run_id,
            "created_files": created_files,
            "created_at": time.time(),
            "protocol_version": PROTOCOL_VERSION,
        }

        manifests = []
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    manifests = json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read rollback manifest at %s; starting fresh.", self.manifest_path)
                manifests = []

        manifests.append(entry)
        os.makedirs(os.path.dirname(self.manifest_path) or ".", exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifests, f, indent=2)
        return entry

    def rollback(self, run_id: str) -> Dict[str, Any]:
        """Delete every file recorded for a run. The rewind button guard_013 requires."""
        if not os.path.exists(self.manifest_path):
            return {"success": False, "error": "No rollback manifest exists", "run_id": run_id}

        with open(self.manifest_path) as f:
            manifests = json.load(f)

        removed, missing = [], []
        remaining = []
        for entry in manifests:
            if entry["run_id"] != run_id:
                remaining.append(entry)
                continue
            for path in entry["created_files"]:
                if os.path.exists(path):
                    os.remove(path)
                    removed.append(path)
                else:
                    missing.append(path)

        with open(self.manifest_path, "w") as f:
            json.dump(remaining, f, indent=2)

        return {"success": True, "run_id": run_id, "removed": removed, "already_absent": missing}

    @staticmethod
    def _escalate(current: str, proposed: str) -> str:
        """Return whichever decision is stricter."""
        return proposed if _DECISION_RANK[proposed] > _DECISION_RANK[current] else current

    def _log(self, decision: PreflightDecision) -> None:
        """Append a decision to the local Runtime Action Log."""
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(decision)) + "\n")
        except OSError as e:
            logger.warning("Could not append to runtime action log: %s", e)


_guard: Optional[BoundaryGuard] = None


def get_boundary_guard() -> BoundaryGuard:
    """Return the process-wide BoundaryGuard instance."""
    global _guard
    if _guard is None:
        _guard = BoundaryGuard()
    return _guard
