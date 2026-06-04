from enum import Enum


class SentimentLabel(str, Enum):
    NEG = "NEG"
    NEU = "NEU"
    POS = "POS"

class TopicLabel(str, Enum):
    CONTENT = "CONTENT"
    TECH = "TECHNICAL"
    POLICY_SECURITY = "POLICY_SECURITY"
    FINANCIAL = "FINANCIAL"


class ActionLabel(str, Enum):
    REPORT_BUG = "REPORT_BUG"
    USER_REQUEST = "USER_REQUEST"
    GENERAL_NOTE = "GENERAL_NOTE"
