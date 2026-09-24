"""Conservative spatial evidence selection for raw OCR.

Independent OCR regions remain unverified observations. The module supports:
- stacked label/value layouts,
- same-row receipt/field values,
- gated multi-column table headers,
- generic same-row text values,
- explicit ambiguity metadata when one label has multiple plausible values,
- question-aware label-vs-value filtering before same-row selection.

No OCR or API calls are performed here.
"""
import re
from difflib import SequenceMatcher

STOPWORDS = {
    'a', 'an', 'the', 'of', 'is', 'are', 'what', 'when', 'where', 'who',
    'which', 'for', 'in', 'on', 'to', 'me', 'my', 'his', 'her', 'their',
    'does', 'do', 'please',
}

MONETARY_TERMS = {
    'amount', 'price', 'cost', 'total', 'subtotal', 'tax',
    'cash', 'change', 'paid', 'payment', 'balance'
}

NUMBER_TERMS = {
    'number', 'reference', 'id', 'identifier', 'code'
}

NAME_TERMS = {
    'name', 'customer', 'person'
}

AMOUNT_PATTERN = re.compile(
    r'^(?:NT\$|[$€£¥])?\s*\d[\d,]*(?:\.\d{1,2})?\s*$'
)

DATE_PATTERN = re.compile(
    r'^\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}$'
)

TIME_PATTERN = re.compile(
    r'^\d{1,2}:\d{2}(?::\d{2})?$'
)

TOKEN_ALIASES = {
    'alternate': 'alternative',
}


def _normalize_token(token):
    token = TOKEN_ALIASES.get(token, token)
    return token


def tokens(text):
    raw = re.findall(r'[a-z0-9]+', str(text).lower())
    return {
        _normalize_token(t)
        for t in raw
        if t not in STOPWORDS
    }


def region_bbox(item):
    box = item.get('bbox')

    if not isinstance(box, dict):
        return None

    try:
        x = float(box['x'])
        y = float(box['y'])
        width = float(box.get('width', box.get('w')))
        height = float(box.get('height', box.get('h')))

        if width <= 0 or height <= 0:
            return None

        return {
            'x': x,
            'y': y,
            'x2': x + width,
            'y2': y + height,
            'width': width,
            'height': height,
        }

    except (ValueError, TypeError, KeyError):
        return None


def horizontal_overlap_fraction(a, b):
    smaller = min(a['width'], b['width'])

    return (
        max(
            0.0,
            min(a['x2'], b['x2']) - max(a['x'], b['x'])
        )
        / smaller
    )


def vertical_overlap_fraction(a, b):
    smaller = min(a['height'], b['height'])

    return (
        max(
            0.0,
            min(a['y2'], b['y2']) - max(a['y'], b['y'])
        )
        / smaller
    )


def _lexical_score(question_tokens, region_text):
    region_tokens = tokens(region_text)

    if not region_tokens:
        return None

    relevant = region_tokens & question_tokens

    if not relevant:
        return None

    query_coverage = len(relevant) / len(question_tokens)
    label_precision = len(relevant) / len(region_tokens)

    # A fuzzy bonus handles very small lexical variants without broadening
    # the match enough to make unrelated OCR text competitive.
    q_phrase = ' '.join(sorted(question_tokens))
    r_phrase = ' '.join(sorted(region_tokens))

    fuzzy = SequenceMatcher(
        None,
        q_phrase,
        r_phrase
    ).ratio()

    return (
        query_coverage,
        label_precision,
        fuzzy,
        len(relevant),
    )


def _same_row_right_neighbors(
    anchor,
    regions,
    numeric_only=False,
):
    ab = region_bbox(anchor)

    result = []

    for candidate in regions:
        if candidate['region_id'] == anchor['region_id']:
            continue

        cb = region_bbox(candidate)

        if cb is None:
            continue

        gap = cb['x'] - ab['x2']

        if gap < 0:
            continue

        overlap = vertical_overlap_fraction(ab, cb)

        if overlap < 0.50:
            continue

        text = str(candidate['text']).strip()

        if numeric_only and not AMOUNT_PATTERN.fullmatch(text):
            continue

        result.append(
            (
                -overlap,
                gap,
                candidate,
                overlap,
            )
        )

    result.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return result[:3]


def _continuation_neighbors(
    anchor,
    primary_neighbors,
    regions,
):
    """
    Keep short stacked continuations in the selected value column.

    This preserves cases such as date+time or two reference numbers.
    They remain unverified candidates; ambiguity is handled separately
    by question type.
    """

    if not primary_neighbors:
        return []

    ab = region_bbox(anchor)

    additions = []

    seen = {
        entry[2]['region_id']
        for entry in primary_neighbors
    }

    for _, _, primary, _ in primary_neighbors:
        pb = region_bbox(primary)

        for candidate in regions:
            cid = candidate['region_id']

            if (
                cid == anchor['region_id']
                or cid in seen
            ):
                continue

            cb = region_bbox(candidate)

            if cb is None:
                continue

            vertical_gap = cb['y'] - pb['y']

            if not (
                0 < vertical_gap
                <= 0.95 * max(
                    ab['height'],
                    pb['height'],
                )
            ):
                continue

            if horizontal_overlap_fraction(pb, cb) < 0.65:
                continue

            # Keep the continuation in the value side of the label.
            if cb['x'] < ab['x2']:
                continue

            additions.append(
                (
                    vertical_gap,
                    candidate,
                )
            )

            seen.add(cid)

    additions.sort(
        key=lambda item: item[0]
    )

    return [
        candidate
        for _, candidate in additions[:3]
    ]


def table_column_neighbors(
    anchor,
    regions,
):
    """
    Find unverified data-row observations beneath a multi-column header.
    """

    ab = region_bbox(anchor)

    if (
        not ab
        or not re.search(
            r'[A-Za-z]',
            str(anchor['text'])
        )
    ):
        return []

    if AMOUNT_PATTERN.fullmatch(
        str(anchor['text']).strip()
    ):
        return []

    header_peers = []

    anchor_center = (
        ab['x'] + ab['x2']
    ) / 2

    for r in regions:
        if r['region_id'] == anchor['region_id']:
            continue

        rb = region_bbox(r)
        txt = str(r['text']).strip()

        if (
            not re.search(
                r'[A-Za-z]',
                txt
            )
            or AMOUNT_PATTERN.fullmatch(txt)
        ):
            continue

        if not (
            -1.5 * ab['height']
            <= rb['y'] - ab['y']
            <= 0.5 * ab['height']
        ):
            continue

        if horizontal_overlap_fraction(ab, rb) > 0.10:
            continue

        center = (
            rb['x'] + rb['x2']
        ) / 2

        if (
            abs(center - anchor_center)
            <= 1.5 * ab['height']
        ):
            continue

        header_peers.append(r)

    distinct_columns = [
        anchor_center
    ]

    for peer in sorted(
        header_peers,
        key=lambda r: region_bbox(r)['x'],
    ):
        pb = region_bbox(peer)

        center = (
            pb['x'] + pb['x2']
        ) / 2

        if all(
            abs(center - existing)
            > 2 * ab['height']
            for existing in distinct_columns
        ):
            distinct_columns.append(center)

    if len(distinct_columns) < 3:
        return []

    header_end = max(
        [ab['y2']]
        + [
            region_bbox(r)['y2']
            for r in header_peers
        ]
    )

    for r in regions:
        rb = region_bbox(r)

        if r['region_id'] == anchor['region_id']:
            continue

        if (
            0
            <= rb['y'] - ab['y']
            <= 2 * ab['height']
            and horizontal_overlap_fraction(
                ab,
                rb,
            ) >= 0.50
            and re.search(
                r'[A-Za-z]',
                str(r['text'])
            )
            and not AMOUNT_PATTERN.fullmatch(
                str(r['text']).strip()
            )
        ):
            header_end = max(
                header_end,
                rb['y2'],
            )

    candidates = []

    for r in regions:
        if r['region_id'] == anchor['region_id']:
            continue

        rb = region_bbox(r)

        if not (
            header_end
            <= rb['y']
            <= header_end + 5 * ab['height']
        ):
            continue

        overlap = horizontal_overlap_fraction(
            ab,
            rb,
        )

        if overlap < 0.50:
            continue

        candidate_center = (
            rb['x'] + rb['x2']
        ) / 2

        if any(
            abs(candidate_center - other)
            < abs(candidate_center - anchor_center)
            for other in distinct_columns
            if other != anchor_center
        ):
            continue

        candidates.append(
            (
                rb['y'],
                -overlap,
                r,
                overlap,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    if not candidates:
        return []

    first_top = candidates[0][0]

    same_row = [
        c
        for c in candidates
        if c[0] - first_top
        <= 2 * ab['height']
    ]

    return [
        (
            r,
            {
                'horizontal_overlap': round(
                    overlap,
                    3,
                ),
                'spatial_direction': 'table_column_below',
                'relationship':
                    'spatial_candidate_not_verified',
            },
        )
        for _, _, r, overlap in same_row[:3]
    ]


def _candidate_is_plausible(
    question,
    text,
):
    q = str(question).casefold()
    raw = str(text).strip()
    q_tokens = tokens(question)

    if (
        'date' in q
        and 'time' not in q
    ):
        return bool(
            DATE_PATTERN.fullmatch(raw)
        )

    if (
        'time' in q
        and 'date' not in q
    ):
        return bool(
            TIME_PATTERN.fullmatch(raw)
        )

    if q_tokens & MONETARY_TERMS:
        return bool(
            AMOUNT_PATTERN.fullmatch(raw)
        )

    if (
        q_tokens & NAME_TERMS
        and 'name' in q_tokens
    ):
        return (
            bool(
                re.search(
                    r'[A-Za-z]',
                    raw,
                )
            )
            and not DATE_PATTERN.fullmatch(raw)
        )

    if q_tokens & NUMBER_TERMS:
        return bool(
            re.search(
                r'[A-Za-z0-9]',
                raw,
            )
        )

    return True


def _ambiguities(
    question,
    relationships,
    evidence_by_region,
    document_id,
):
    groups = {}

    for relation in relationships:
        groups.setdefault(
            relation['label_region_id'],
            [],
        ).append(relation)

    ambiguous = []

    for label_id, relations in groups.items():
        plausible = []
        seen_text = set()

        for relation in relations:
            cid = relation[
                'candidate_region_id'
            ]

            candidate = evidence_by_region.get(
                cid
            )

            if not candidate:
                continue

            text = str(
                candidate.get(
                    'text',
                    '',
                )
            ).strip()

            if not _candidate_is_plausible(
                question,
                text,
            ):
                continue

            normalized = text.casefold()

            if normalized in seen_text:
                continue

            seen_text.add(normalized)

            plausible.append(
                {
                    'region_id': cid,
                    'text': text,
                    'spatial_direction':
                        relation.get(
                            'spatial_direction'
                        ),
                }
            )

        if len(plausible) > 1:
            ambiguous.append(
                {
                    'document_id':
                        document_id,
                    'label_region_id':
                        label_id,
                    'candidate_region_ids':
                        [
                            p['region_id']
                            for p in plausible
                        ],
                    'candidate_count':
                        len(plausible),
                    'reason':
                        'multiple_plausible_spatial_values',
                }
            )

    return ambiguous


def expand_raw_ocr(
    question,
    document,
    max_anchors=1,
):
    """
    Select matching anchors and nearby OCR candidates,
    preserving IDs.
    """

    if document.get(
        'document_type'
    ) != 'unclassified':
        return [], []

    regions = [
        r
        for r in document.get(
            'evidence',
            [],
        )
        if (
            isinstance(r, dict)
            and r.get(
                'evidence_type'
            ) == 'ocr_region'
            and r.get(
                'meaningful'
            ) is not False
            and r.get('text')
            and region_bbox(r)
            is not None
        )
    ]

    query_tokens = tokens(question)

    if not query_tokens:
        return [], []

    monetary = (
        bool(
            query_tokens
            & MONETARY_TERMS
        )
        and not bool(
            query_tokens
            & {
                'date',
                'time',
            }
        )
    )

    matches = []

    for region in regions:
        lexical = _lexical_score(
            query_tokens,
            region['text'],
        )

        if lexical is None:
            continue

        (
            query_coverage,
            label_precision,
            fuzzy,
            relevant_count,
        ) = lexical

        if query_coverage < 0.5:
            continue

        # Prefer a field-like anchor that already has a
        # same-row value. This resolves section-heading
        # ties such as CUSTOMER vs Name.
        row_candidates = (
            _same_row_right_neighbors(
                region,
                regions,
                numeric_only=monetary,
            )
        )

        # Experiment 61:
        # A nearby OCR region is not automatically a value.
        # Filter same-row candidates using the question type
        # before using geometry as an anchor-selection bonus.
        # This prevents another table header such as
        # "Card Pick-up" from making "Payment Date" look like
        # it already has a same-row value.
        row_candidates = [
            entry
            for entry in row_candidates
            if _candidate_is_plausible(
                question,
                entry[2].get(
                    'text',
                    '',
                ),
            )
        ]

        geometry_bonus = (
            1
            if row_candidates
            else 0
        )

        matches.append(
            (
                query_coverage,
                label_precision,
                geometry_bonus,
                fuzzy,
                relevant_count,
                region,
            )
        )

    def _region_order(region):
        try:
            return int(
                region.get(
                    'source_order',
                    region.get(
                        'region_id',
                        0,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0

    matches.sort(
        key=lambda v: (
            -v[0],
            -v[1],
            -v[2],
            -v[3],
            -v[4],
            _region_order(v[5]),
        )
    )

    chosen = []
    relationships = []
    seen = set()

    for (
        _,
        _,
        _,
        _,
        _,
        anchor,
    ) in matches[:max_anchors]:
        aid = anchor['region_id']

        if aid not in seen:
            chosen.append(anchor)
            seen.add(aid)

        ab = region_bbox(anchor)

        vertical_neighbors = []

        for candidate in regions:
            if candidate[
                'region_id'
            ] == aid:
                continue

            cb = region_bbox(candidate)

            vertical = (
                cb['y']
                - ab['y']
            )

            horizontal_overlap = (
                horizontal_overlap_fraction(
                    ab,
                    cb,
                )
            )

            if (
                horizontal_overlap >= 0.50
                and 0
                <= vertical
                <= 1.60 * ab['height']
            ):
                vertical_neighbors.append(
                    (
                        vertical,
                        -horizontal_overlap,
                        candidate,
                        horizontal_overlap,
                    )
                )

        # Generic same-row values are allowed
        # for any field-like question.
        # Monetary questions retain the
        # numeric-only restriction.
        same_row = (
            _same_row_right_neighbors(
                anchor,
                regions,
                numeric_only=monetary,
            )
        )

        # Experiment 61:
        # reject same-row observations that do
        # not look like a plausible value for
        # the question. If every same-row
        # observation is label-like, the existing
        # table-column-below logic gets a chance
        # to resolve the value instead.
        same_row = [
            entry
            for entry in same_row
            if _candidate_is_plausible(
                question,
                entry[2].get(
                    'text',
                    '',
                ),
            )
        ]

        continuations = (
            _continuation_neighbors(
                anchor,
                same_row,
                regions,
            )
        )

        column_neighbors = (
            table_column_neighbors(
                anchor,
                regions,
            )
        )

        if same_row:
            neighbors = [
                (
                    candidate,
                    {
                        'vertical_overlap':
                            round(
                                overlap,
                                3,
                            ),
                        'horizontal_gap':
                            round(
                                gap,
                                3,
                            ),
                        'spatial_direction':
                            'same_row_right',
                    },
                )
                for (
                    _,
                    gap,
                    candidate,
                    overlap,
                ) in same_row
            ]

            for candidate in continuations:
                cb = region_bbox(
                    candidate
                )

                neighbors.append(
                    (
                        candidate,
                        {
                            'horizontal_overlap':
                                round(
                                    horizontal_overlap_fraction(
                                        region_bbox(
                                            same_row[0][2]
                                        ),
                                        cb,
                                    ),
                                    3,
                                ),
                            'spatial_direction':
                                'same_row_right_continuation',
                        },
                    )
                )

        elif column_neighbors:
            neighbors = [
                (
                    candidate,
                    {
                        k: v
                        for k, v
                        in details.items()
                        if k
                        != 'relationship'
                    },
                )
                for (
                    candidate,
                    details,
                ) in column_neighbors
            ]

        else:
            vertical_neighbors.sort(
                key=lambda v: (
                    v[0],
                    v[1],
                )
            )

            neighbors = [
                (
                    candidate,
                    {
                        'horizontal_overlap':
                            round(
                                overlap,
                                3,
                            ),
                        'spatial_direction':
                            'below',
                    },
                )
                for (
                    _,
                    _,
                    candidate,
                    overlap,
                ) in vertical_neighbors[:3]
            ]

        for (
            candidate,
            details,
        ) in neighbors:
            cid = candidate[
                'region_id'
            ]

            relationships.append(
                {
                    'label_region_id':
                        aid,
                    'candidate_region_id':
                        cid,
                    **details,
                    'relationship':
                        'spatial_candidate_not_verified',
                }
            )

            if cid not in seen:
                chosen.append(candidate)
                seen.add(cid)

    return chosen, relationships


def build_raw_ocr_packet(
    question,
    documents,
):
    evidence = []
    relations = []
    ambiguities = []

    for document in documents:
        regions, candidates = (
            expand_raw_ocr(
                question,
                document,
            )
        )

        did = document[
            'document_id'
        ]

        region_lookup = {
            r['region_id']: r
            for r in document.get(
                'evidence',
                [],
            )
            if (
                isinstance(r, dict)
                and r.get(
                    'region_id'
                ) is not None
            )
        }

        for region in regions:
            evidence.append(
                {
                    'evidence_item_id':
                        f"{did}:ocr_region:{region['region_id']}",
                    'document_id':
                        did,
                    'item_type':
                        'ocr_region',
                    'content':
                        {
                            'region_id':
                                region['region_id'],
                            'text':
                                region['text'],
                            'bbox':
                                region['bbox'],
                            'confidence':
                                region.get(
                                    'confidence'
                                ),
                        },
                }
            )

        doc_relations = [
            dict(
                document_id=did,
                **relation,
            )
            for relation in candidates
        ]

        relations.extend(
            doc_relations
        )

        ambiguities.extend(
            _ambiguities(
                question,
                candidates,
                region_lookup,
                did,
            )
        )

    return {
        'question':
            question,

        'answer_mode':
            (
                'raw_ocr_spatial_candidates'
                if evidence
                else 'no_answer'
            ),

        'evidence':
            evidence,

        'spatial_candidates':
            relations,

        'ambiguities':
            ambiguities,

        'evidence_level':
            'raw_ocr_unverified_relationships',
    }