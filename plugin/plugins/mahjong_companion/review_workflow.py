from __future__ import annotations

from pathlib import Path
from typing import Any

from .review import (
    append_review_summary_history,
    build_review_summary,
    generate_coaching_topics,
    generate_coaching_trend,
    generate_review_summary as generate_review_summary_artifact,
    load_review_candidates,
    sync_memory_bridge_queue,
)
from .session_state import now_iso


class ReviewWorkflowMixin:
    def _generate_review_summary_locked(self) -> dict[str, Any]:
        cache_dir = self.plugin.data_path("session_cache")
        try:
            summary, summary_path = generate_review_summary_artifact(
                cache_dir,
                session_id=self.state.session_id,
            )
            history_path = append_review_summary_history(
                cache_dir,
                summary,
                limit=int(self._get_coaching_cfg().get("history_limit", 24)),
            )
            payload = self._apply_review_summary_result(summary)
            payload.update(self._refresh_coaching_state_locked(cache_dir))
            payload["ok"] = True
            payload["path"] = str(summary_path)
            payload["history_path"] = str(history_path)
            self._emit_status()
            return payload
        except Exception as exc:
            self.logger.exception("generate_review_summary failed")
            self._mark_review_summary_failure(str(exc))
            return {
                "ok": False,
                "error": str(exc),
            }

    def _generate_review_summary_from_file_locked(self, review_candidates_path: str) -> dict[str, Any]:
        candidate = self._resolve_user_review_candidates_path(review_candidates_path)
        if isinstance(candidate, dict):
            self._mark_review_summary_failure(str(candidate.get("error", "invalid review candidates path")))
            candidate["source_path"] = str(review_candidates_path)
            return candidate
        try:
            items = load_review_candidates(candidate)
            summary = build_review_summary(
                session_id=self.state.session_id,
                candidates=items,
            )
            cache_dir = self.plugin.data_path("session_cache")
            history_path = append_review_summary_history(
                cache_dir,
                summary,
                limit=int(self._get_coaching_cfg().get("history_limit", 24)),
            )
            payload = self._apply_review_summary_result(summary)
            payload.update(self._refresh_coaching_state_locked(cache_dir))
            payload["ok"] = True
            payload["source_path"] = str(candidate)
            payload["history_path"] = str(history_path)
            self._emit_status()
            return payload
        except Exception as exc:
            self.logger.exception("generate_review_summary_from_file failed")
            self._mark_review_summary_failure(str(exc))
            return {
                "ok": False,
                "error": str(exc),
                "source_path": str(candidate),
            }

    def _sync_memory_bridge_locked(self) -> dict[str, Any]:
        cache_dir = self.plugin.data_path("session_cache")
        bridge_cfg = self._get_memory_bridge_cfg()
        report, report_path = sync_memory_bridge_queue(
            cache_dir,
            writer=self._host_memory_writer,
            bucket_id=str(bridge_cfg.get("host_memory_bucket_id", "mahjong_companion_coaching")),
            batch_size=int(bridge_cfg.get("host_sync_batch_size", 5)),
        )
        self._apply_host_memory_sync_result(report)
        self._emit_status()
        payload = dict(report)
        payload["path"] = str(report_path)
        return payload

    def _get_coaching_trend_locked(self) -> dict[str, Any]:
        cache_dir = self.plugin.data_path("session_cache")
        try:
            payload = self._refresh_coaching_state_locked(cache_dir)
            payload["ok"] = True
            self._emit_status()
            return payload
        except Exception as exc:
            self.state.last_error = str(exc)
            self._emit_status()
            return {
                "ok": False,
                "error": str(exc),
            }

    def _get_last_coaching_topics_locked(self) -> dict[str, Any]:
        if not self.state.last_coaching_topics:
            cache_dir = self.plugin.data_path("session_cache")
            try:
                self._refresh_coaching_state_locked(cache_dir)
            except Exception as exc:
                self.state.last_error = str(exc)
                self._emit_status()
                return {
                    "ok": False,
                    "error": str(exc),
                    "topics": [],
                }
        return {
            "ok": bool(self.state.last_coaching_topics),
            "coach_focus": self.state.last_coaching_focus,
            "summary_text": self.state.last_coaching_summary_text,
            "topics": list(self.state.last_coaching_topics),
            "last_coaching_trend_at": self.state.last_coaching_trend_at,
        }

    def _apply_review_summary_result(self, summary: dict[str, Any]) -> dict[str, Any]:
        self.state.last_review_summary_at = now_iso()
        self.state.last_review_summary_ok = True
        self.state.last_review_summary = dict(summary)
        self.state.last_review_summary_text = str(summary.get("summary_text", ""))
        self.state.last_error = ""
        return dict(summary)

    def _apply_host_memory_sync_result(self, report: dict[str, Any]) -> dict[str, Any]:
        self.state.last_host_memory_sync_at = str(report.get("attempted_at") or now_iso())
        self.state.last_host_memory_sync_status = str(report.get("status", ""))
        self.state.last_host_memory_sync_note = str(report.get("note", ""))
        self.state.last_host_memory_sync_pending = int(report.get("pending_count", 0) or 0)
        self.state.last_error = ""
        return dict(report)

    def _apply_coaching_outputs(self, trend: dict[str, Any], topics_payload: dict[str, Any]) -> dict[str, Any]:
        self.state.last_coaching_trend_at = str(trend.get("generated_at") or now_iso())
        self.state.last_coaching_trend = dict(trend)
        self.state.last_coaching_summary_text = str(trend.get("summary_text", ""))
        self.state.last_coaching_focus = str(trend.get("coach_focus", ""))
        topics = topics_payload.get("topics", [])
        self.state.last_coaching_topics = list(topics) if isinstance(topics, list) else []
        self.state.last_error = ""
        return {
            "coaching_trend": dict(trend),
            "coaching_topics": dict(topics_payload),
        }

    def _refresh_coaching_state_locked(self, cache_dir: Path) -> dict[str, Any]:
        coaching_cfg = self._get_coaching_cfg()
        trend, trend_path = generate_coaching_trend(
            cache_dir,
            session_window=int(coaching_cfg.get("trend_window_sessions", 3)),
        )
        topics_payload, topics_path = generate_coaching_topics(
            cache_dir,
            trend,
            topic_limit=int(coaching_cfg.get("topic_limit", 3)),
        )
        payload = self._apply_coaching_outputs(trend, topics_payload)
        payload["coaching_trend_path"] = str(trend_path)
        payload["coaching_topics_path"] = str(topics_path)
        return payload
