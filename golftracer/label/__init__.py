"""OpenCV label application and portable label schema."""

from .schema import Label, LabelDocument, load_labels, save_labels

__all__ = ["Label", "LabelDocument", "load_labels", "save_labels"]
