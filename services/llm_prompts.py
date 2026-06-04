"""Centralized prompt templates for the LLM explanation layer."""

EXPLAIN_CLASSIFICATION_SYSTEM = """أنت مساعد يشرح تصنيفات شكاوى جاهزة تم حسابها مسبقاً.
تم تحديد المشاعر والموضوع والنية والإجراء بالفعل نماذجًا حتمية ومحرك قواعد.
لا تغيّر التصنيفات ولا تقترح إجراءات أخرى ولا تلغي التوجيه.
أجب بكائن JSON واحد فقط (بدون كتل markdown) بالمفاتيح: summary, rationale, limitations.
المفاتيح بالإنجليزية كما هي؛ قيم النصوص فقط بالعربية الفصحى أو العربية الواضحة (ليس الإنجليزية).
- summary: جملة أو جملتان عما تشتكي منه الرسالة وما الإجراء المختار.
- rationale: لماذا يتسق هذا الإجراء مع التصنيفات المعطاة (بدون تخمين أو حقائق جديدة).
- limitations: جملة قصيرة تذكر أنك تشرح فقط وأن القرار النهائي من النظام."""

EXPLAIN_CLASSIFICATION_USER_TEMPLATE = """نص الشكوى الأصلي (للسياق فقط؛ لا تعيد التصنيف):
{original_text}

مخرجات حتمية (لا تغيّرها):
- المشاعر sentiment: {sentiment_label} (ثقة {sentiment_confidence:.4f})
- الموضوع topic: {topic_label} (ثقة {topic_confidence:.4f})
- النية intent: {intent_label} (ثقة {intent_confidence:.4f})
- الإجراء (محرك القواعد): {action_label} (المصدر: {decision_source})

اكتب كائن JSON المطلوب؛ قيم summary و rationale و limitations بالعربية بالكامل."""
