penalty_mapping = {
    # Орфографічні помилки (Spelling and Punctuation)
    "Spelling": 0.5,
    "Punctuation": 0.3,

    # Граматичні помилки (Grammatical Errors)
    "G/Case": 1.0,
    "G/Gender": 1.0,
    "G/Number": 0.8,
    "G/Aspect": 0.9,
    "G/Tense": 0.9,
    "G/VerbVoice": 0.9,
    "G/PartVoice": 0.9,
    "G/VerbAForm": 0.8,
    "G/Prep": 0.7,
    "G/Participle": 0.8,
    "G/UngrammaticalStructure": 1.2,
    "G/Comparison": 0.7,
    "G/Conjunction": 0.6,
    "G/Other": 0.8,

    # Мовні помилки (Linguistic Errors)
    "F/Style": 0.5,
    "F/Calque": 0.7,
    "F/Collocation": 0.6,
    "F/PoorFlow": 0.7,
    "F/Repetition": 0.4,
    "F/Other": 0.5
}
