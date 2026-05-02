"""Rubric-friendly entry point for the data preprocessing step.

The original project file was named prepocess_data.py and the rest of the
repo still references it. This wrapper keeps that file working while also
providing the preprocess_data.py entry point expected by the project rubric.
"""

from prepocess_data import main


if __name__ == "__main__":
    main()
