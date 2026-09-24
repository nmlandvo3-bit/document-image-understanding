"""
Document QA Service

Reusable application service built from the validated
Experiments 40-45.

Pipeline
--------
Unified documents
    -> generic retrieval
    -> evidence compression
    -> grounded Groq request
    -> provenance validation
    -> application response

This module does NOT rerun OCR or document parsing.
"""

import json
import os
import sys

from groq import Groq

from experiments.experiment_41_generic_cross_document_retrieval import (
    PIC1_PATH,
    PIC13_PATH,
    build_search_corpus,
    load_json,
)

from experiments.experiment_44_grounding_evidence_compression import (
    build_compressed_packet,
)


if hasattr(
    sys.stdout,
    "reconfigure",
):
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass


DEFAULT_MODEL = (
    "openai/gpt-oss-20b"
)


SYSTEM_PROMPT = """
You answer questions about documents.

Rules:
- Use only the supplied document evidence.
- Do not use outside knowledge.
- Do not invent missing facts.
- Do not silently correct OCR.
- If evidence is insufficient, say so.
- Use only supplied evidence_item_ids.
- Keep the answer concise.

Return JSON only:

{
  "answer": "natural-language answer",
  "evidence_item_ids": ["ids actually used"],
  "uncertainty": null
}
""".strip()


class DocumentQAService:
    def __init__(
        self,
        model=DEFAULT_MODEL,
        api_key=None,
    ):
        self.model = model

        self.api_key = (
            api_key
            or os.getenv(
                "GROQ_API_KEY"
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set."
            )

        self.client = Groq(
            api_key=self.api_key
        )

        self.documents = []
        self.corpus = []

        self._load_documents()

    def _load_documents(
        self,
    ):
        """
        Load validated unified documents.
        """

        paths = [
            PIC1_PATH,
            PIC13_PATH,
        ]

        documents = []

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

            documents.append(
                load_json(
                    path
                )
            )

        self.documents = documents

        self.corpus = (
            build_search_corpus(
                self.documents
            )
        )

    def get_document_summary(
        self,
    ):
        """
        Lightweight information about indexed documents.
        """

        summary = []

        for document in self.documents:
            summary.append(
                {
                    "document_id": (
                        document.get(
                            "document_id"
                        )
                    ),
                    "document_type": (
                        document.get(
                            "document_type"
                        )
                    ),
                    "records": len(
                        document.get(
                            "records",
                            [],
                        )
                    ),
                    "fields": len(
                        document.get(
                            "fields",
                            [],
                        )
                    ),
                    "controls": len(
                        document.get(
                            "controls",
                            [],
                        )
                    ),
                    "free_text": len(
                        document.get(
                            "free_text",
                            [],
                        )
                    ),
                }
            )

        return summary

    @staticmethod
    def _compact_content(
        evidence_item,
    ):
        item_type = (
            evidence_item[
                "item_type"
            ]
        )

        content = (
            evidence_item[
                "content"
            ]
        )

        if item_type == "record":
            return {
                "record_id": (
                    content.get(
                        "record_id"
                    )
                ),
                "category": (
                    content.get(
                        "category"
                    )
                ),
                "event": (
                    content.get(
                        "event"
                    )
                ),
                "time": (
                    content.get(
                        "time"
                    )
                ),
                "detail": (
                    content.get(
                        "detail"
                    )
                ),
                "explanation": (
                    content.get(
                        "explanation"
                    )
                ),
                "review_flags": (
                    content.get(
                        "review_flags",
                        [],
                    )
                ),
            }

        if item_type == "field":
            return {
                "label": (
                    content.get(
                        "label"
                    )
                ),
                "value": (
                    content.get(
                        "value"
                    )
                ),
            }

        if item_type == "control":
            return {
                "label": (
                    content.get(
                        "label"
                    )
                ),
                "checked": bool(
                    content.get(
                        "checked",
                        False,
                    )
                ),
                "state": (
                    content.get(
                        "state"
                    )
                ),
            }

        if item_type == "free_text":
            return {
                "source_control_label": (
                    content.get(
                        "source_control_label"
                    )
                ),
                "text": (
                    content.get(
                        "text"
                    )
                ),
            }

        return content

    def build_llm_packet(
        self,
        question,
    ):
        """
        Build compressed evidence for the LLM.
        """

        packet = (
            build_compressed_packet(
                question,
                self.corpus,
                top_k=15,
            )
        )

        compact_evidence = []

        for item in packet.get(
            "evidence",
            [],
        ):
            compact_evidence.append(
                {
                    "evidence_item_id": (
                        item[
                            "item_id"
                        ]
                    ),
                    "document_id": (
                        item[
                            "document_id"
                        ]
                    ),
                    "item_type": (
                        item[
                            "item_type"
                        ]
                    ),
                    "content": (
                        self._compact_content(
                            item
                        )
                    ),
                }
            )

        return {
            "question": question,
            "answer_mode": (
                packet.get(
                    "answer_mode"
                )
            ),
            "evidence": (
                compact_evidence
            ),
        }

    @staticmethod
    def _parse_json_response(
        content,
    ):
        content = (
            content.strip()
        )

        if content.startswith(
            "```"
        ):
            lines = (
                content.splitlines()
            )

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1]
                .strip()
                .startswith(
                    "```"
                )
            ):
                lines = (
                    lines[:-1]
                )

            content = "\n".join(
                lines
            ).strip()

        return json.loads(
            content
        )

    def _call_groq(
        self,
        packet,
    ):
        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": (
                            "system"
                        ),
                        "content": (
                            SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            json.dumps(
                                packet,
                                ensure_ascii=False,
                                separators=(
                                    ",",
                                    ":",
                                ),
                            )
                        ),
                    },
                ],
                temperature=0.0,
                reasoning_effort="low",
                include_reasoning=False,
                response_format={
                    "type": (
                        "json_object"
                    ),
                },
                max_completion_tokens=400,
                stream=False,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        parsed = (
            self._parse_json_response(
                content
            )
        )

        required = {
            "answer",
            "evidence_item_ids",
            "uncertainty",
        }

        missing = (
            required
            - set(
                parsed.keys()
            )
        )

        if missing:
            raise RuntimeError(
                "LLM response missing: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        usage = {}

        if getattr(
            response,
            "usage",
            None,
        ):
            usage = {
                "prompt_tokens": getattr(
                    response.usage,
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    response.usage,
                    "completion_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    response.usage,
                    "total_tokens",
                    None,
                ),
            }

        return {
            "answer": str(
                parsed[
                    "answer"
                ]
            ),
            "evidence_item_ids": list(
                parsed[
                    "evidence_item_ids"
                ]
            ),
            "uncertainty": (
                parsed[
                    "uncertainty"
                ]
            ),
            "usage": usage,
        }

    @staticmethod
    def _validate_evidence(
        packet,
        llm_result,
    ):
        supplied_ids = {
            item[
                "evidence_item_id"
            ]
            for item in packet[
                "evidence"
            ]
        }

        used_ids = set(
            llm_result[
                "evidence_item_ids"
            ]
        )

        invalid_ids = (
            used_ids
            - supplied_ids
        )

        return {
            "valid": (
                len(
                    invalid_ids
                )
                == 0
            ),
            "supplied_ids": (
                sorted(
                    supplied_ids
                )
            ),
            "used_ids": (
                sorted(
                    used_ids
                )
            ),
            "invalid_ids": (
                sorted(
                    invalid_ids
                )
            ),
        }

    def answer(
        self,
        question,
    ):
        """
        Main application-facing method.
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question must be a string."
            )

        question = (
            question.strip()
        )

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        packet = (
            self.build_llm_packet(
                question
            )
        )

        if not packet[
            "evidence"
        ]:
            return {
                "question": question,
                "answer": (
                    "I could not find sufficient "
                    "document evidence to answer "
                    "that question."
                ),
                "grounded": False,
                "provider": "groq",
                "model": self.model,
                "answer_mode": (
                    packet.get(
                        "answer_mode"
                    )
                ),
                "evidence": [],
                "evidence_validation": {
                    "valid": True,
                    "supplied_ids": [],
                    "used_ids": [],
                    "invalid_ids": [],
                },
                "uncertainty": (
                    "insufficient_document_evidence"
                ),
                "usage": {},
            }

        llm_result = (
            self._call_groq(
                packet
            )
        )

        validation = (
            self._validate_evidence(
                packet,
                llm_result,
            )
        )

        if not validation[
            "valid"
        ]:
            raise RuntimeError(
                "LLM returned evidence IDs "
                "that were not supplied by "
                "the document pipeline."
            )

        evidence_by_id = {
            item[
                "evidence_item_id"
            ]: item
            for item in packet[
                "evidence"
            ]
        }

        used_evidence = []

        for evidence_id in (
            llm_result[
                "evidence_item_ids"
            ]
        ):
            item = (
                evidence_by_id.get(
                    evidence_id
                )
            )

            if item is not None:
                used_evidence.append(
                    item
                )

        return {
            "question": question,
            "answer": (
                llm_result[
                    "answer"
                ]
            ),
            "grounded": True,
            "provider": "groq",
            "model": self.model,
            "answer_mode": (
                packet.get(
                    "answer_mode"
                )
            ),
            "evidence": (
                used_evidence
            ),
            "evidence_validation": (
                validation
            ),
            "uncertainty": (
                llm_result[
                    "uncertainty"
                ]
            ),
            "usage": (
                llm_result[
                    "usage"
                ]
            ),
        }


def interactive_cli():
    print(
        "=" * 72
    )

    print(
        "DOCUMENT QA SERVICE"
    )

    print(
        "=" * 72
    )

    service = (
        DocumentQAService()
    )

    print(
        "Indexed documents:"
    )

    for document in (
        service
        .get_document_summary()
    ):
        print(
            (
                f"  {document['document_id']} "
                f"({document['document_type']})"
            )
        )

    print()

    print(
        "Type a question."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:
        try:
            question = input(
                "Question> "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        if not question:
            continue

        try:
            result = (
                service.answer(
                    question
                )
            )

            print()
            print(
                "Answer:"
            )

            print(
                result[
                    "answer"
                ]
            )

            print()

            print(
                "Grounded:",
                result[
                    "grounded"
                ],
            )

            print(
                "Evidence:"
            )

            if not result[
                "evidence"
            ]:
                print(
                    "  None"
                )

            for item in result[
                "evidence"
            ]:
                print(
                    " ",
                    item[
                        "evidence_item_id"
                    ],
                )

            print()

        except Exception as exc:
            print()
            print(
                (
                    f"ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            print()


if __name__ == "__main__":
    interactive_cli()