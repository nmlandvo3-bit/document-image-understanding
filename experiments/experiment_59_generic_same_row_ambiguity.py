"""Experiment 59: generic same-row retrieval and ambiguity inspection.

Run after installing the Experiment 59 retrieval module. This reads an existing
unified_raw_ocr.json only. It does not run OCR or call Groq.
"""
import argparse
import json
from pathlib import Path

from services.layout_aware_ocr_retrieval import build_raw_ocr_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', required=True)
    args = parser.parse_args()

    path = Path(args.input_json)
    document = json.loads(path.read_text(encoding='utf-8'))

    questions = [
        'what is the payment date?',
        'what is the Alternate Date?',
        'what is the Reference Number?',
        'what is the Customer Name?',
        'what is the total?',
        'what is the Tax?',
        'what is the Cash?',
        'what is the Date of Birth?',
    ]

    print('=' * 76)
    print('EXPERIMENT 59: GENERIC SAME-ROW + AMBIGUITY')
    print('DOCUMENT:', document.get('document_id'))
    print('=' * 76)

    for question in questions:
        packet = build_raw_ocr_packet(question, [document])
        print('\nQUESTION:', question)
        print('ANSWER MODE:', packet.get('answer_mode'))
        print('EVIDENCE:')
        if not packet.get('evidence'):
            print('  None')
        for item in packet.get('evidence', []):
            content = item.get('content', {})
            print(f"  Region {content.get('region_id')}: {content.get('text')!r}")
        print('AMBIGUITIES:', packet.get('ambiguities', []))

    print('\n' + '=' * 76)
    print('EXPERIMENT 59 INSPECTION COMPLETE')


if __name__ == '__main__':
    main()
