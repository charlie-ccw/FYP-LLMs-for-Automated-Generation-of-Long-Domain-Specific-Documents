GENERATION_WITH_TEMPLATE_AND_KEY_INFO_SYSTEM = "You are a document writer"
GENERATION_WITH_TEMPLATE_AND_KEY_INFO_PROMPT = """Your task is to write document for the current section based on the provided section requirement.
You can use Key Info to get extra knowledge to make your document more professional and comprehensive.
<Section Requirement>
{section_requirement}
</Section Requirement>

<Key Info>
{key_info}
</Key Info>

Note: Please optimize the generated content as much as possible. You can improve it from various aspects such as overall structure and layout, content richness, and fluency."""