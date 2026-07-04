# -*- coding: utf-8 -*-
"""Piedmont Healthcare Analysis — end-to-end pipeline on real Synthea data.

Run this single script to reproduce the full report:
    python piedmont_main.py
"""

import piedmont_eda as eda
import piedmont_ml_highcost as highcost_model
import piedmont_ml_procedure_cost as procedure_cost_model
import piedmont_rfm as rfm


def main() -> None:
    data = eda.run()
    rfm.run(data["patients"], data["encounters"])
    print()
    highcost_model.run()
    print()
    procedure_cost_model.run()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Charts saved to: piedmont_output/")
    print("  01_eda_summary.png")
    print("  02_rfm_segmentation.png")
    print("  03_highcost_classifier.png")
    print("  04_procedure_cost_model.png")


if __name__ == "__main__":
    main()
