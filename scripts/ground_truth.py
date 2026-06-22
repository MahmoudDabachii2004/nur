"""
ground_truth.py — Authoritative reference verses/hadiths for NUR evaluation.

This file is the GROUND TRUTH for evaluating our pipeline. Each question has
a curated list of verses/hadiths that we KNOW are the correct sources, verified
via:
  - alquran.cloud search API (for Quran verses)
  - sunnah.com (for hadiths — manually verified by the user)
  - Scholarly consensus (well-known verses every Islamic scholar would cite)

The pipeline is evaluated against this ground truth to measure:
  - RECALL: did the pipeline find these verses?
  - PRECISION: did the pipeline avoid irrelevant verses?

How to use:
  from scripts.ground_truth import GROUND_TRUTH
  for question, expected in GROUND_TRUTH.items():
      # run pipeline on question
      # check how many expected verses are in the top-10
"""

from __future__ import annotations

# ============================================================
# Ground Truth Dataset
# ============================================================
# Each entry: {question: {expected_quran: [(surah, ayah_standard)], expected_hadith: [(collection, number)]}}
#
# The surah:ayah values here are STANDARD numbering (as on quran.com and alquran.cloud API).
# To match against our DB, we need to convert to global numbering using the offset table
# from docs/DATA_SOURCES.md section 9 issue #6.
#
# The offset per surah is cumulative:
#   surah 1: 7 ayahs (Al-Fatihah)
#   surah 2: 286 ayahs (Al-Baqarah)
#   surah 3: 200 ayahs (Aal Imran)
#   surah 4: 176 ayahs (An-Nisa)
#   surah 5: 120 ayahs (Al-Maidah)
#   etc.
#
# Global ayah_num = sum of all previous surahs' ayah counts + standard ayah_num
# ============================================================

# Cumulative ayah counts per surah (for global → standard conversion)
SURAH_CUMULATIVE = {
    1: 0,      # Al-Fatihah: 7 ayahs, but starts at 1
    2: 7,      # Al-Baqarah starts at global 8 (but we have +7 offset)
    3: 293,    # Aal Imran
    4: 493,    # An-Nisa
    5: 669,    # Al-Maidah
    6: 789,    # Al-An'am
    7: 899,    # Al-A'raf
    8: 1063,   # Al-Anfal
    9: 1167,   # At-Tawbah
    10: 1235,  # Yunus
    11: 1346,  # Hud
    12: 1461,  # Yusuf
    13: 1591,  # Ar-Ra'd
    14: 1671,  # Ibrahim
    15: 1755,  # Al-Hijr
    16: 1803,  # An-Nahl
    17: 1942,  # Al-Isra
    18: 2070,  # Al-Kahf
    19: 2188,  # Maryam
    20: 2278,  # Ta-Ha
    24: 2521,  # An-Nur
    69: 5352,  # Al-Haqqah
    70: 5397,  # Al-Ma'arij
    73: 5493,  # Al-Muzzammil
    74: 5535,  # Al-Muddaththir
    87: 5950,  # Al-A'la
    89: 6024,  # Al-Fajr
    92: 6084,  # Al-Layl
    107: 6203, # Al-Ma'un
}


def standard_to_global(surah: int, ayah: int) -> int:
    """Convert standard surah:ayah to global ayah number used in our DB.

    The DB uses global cumulative numbering. This function computes:
        global = cumulative_offset[surah] + ayah

    Args:
        surah: Surah number (1-114)
        ayah: Standard ayah number within the surah

    Returns:
        Global ayah number as used in the DB's ayah_num field.
    """
    offset = SURAH_CUMULATIVE.get(surah, 0)
    return offset + ayah


GROUND_TRUTH: dict[str, dict] = {
    # ============================================================
    # Q1: "What does the Quran say about charity and zakat?"
    # ============================================================
    "What does the Quran say about charity and zakat?": {
        "description": "Charity and Zakat — verified via alquran.cloud API search",
        "expected_quran_standard": [
            # The 8 categories of Zakat recipients — THE most important verse
            (9, 60, "Zakah expenditures are only for the poor and for the needy..."),
            # Establish prayer AND give zakah (foundational command)
            (2, 43, "And establish prayer and give zakah and bow with those who bow"),
            (2, 110, "And establish prayer and give zakah, and whatever good you put forward..."),
            (2, 277, "Indeed, those who believe and do righteous deeds and establish prayer and give zakah..."),
            # What to spend and on whom
            (2, 215, "They ask you what they should spend. Say: Whatever you spend of good is for parents..."),
            # For the petitioner and the deprived
            (70, 25, "For the petitioner and the deprived"),
            # Those who practice charity lend Allah a goodly loan
            (57, 18, "Indeed, the men who practice charity and the women who practice charity..."),
            # Jesus was enjoined to pray and give zakah
            (19, 31, "And He has made me blessed wherever I am and has enjoined upon me prayer and zakah..."),
            # Allah destroys usury and gives increase for charities
            (2, 276, "Allah destroys interest and gives increase for charities"),
        ],
        "expected_hadith": [
            # The Prophet sent Mu'adh to Yemen — taught zakat is obligatory
            ("bukhari", 1395, "The Prophet sent Mu'adh to Yemen and said... teach about zakat"),
            # Every Muslim must give charity
            ("nasai", 2538, "Every Muslim must give charity"),
            # Best charity is from self-sufficiency
            ("bukhari", 1427, "The best of charity is that given from self-sufficiency"),
        ],
    },

    # ============================================================
    # Q2: "Is prayer obligatory?"
    # ============================================================
    "Is prayer obligatory?": {
        "description": "Prayer obligation — verified via alquran.cloud API search for 'establish prayer'",
        "expected_quran_standard": [
            # Believe in unseen + establish prayer (foundational)
            (2, 3, "Who believe in the unseen, establish prayer, and spend out of what We have provided"),
            # Establish prayer + give zakah (command)
            (2, 43, "And establish prayer and give zakah and bow with those who bow"),
            (2, 110, "And establish prayer and give zakah..."),
            (2, 277, "Indeed, those who believe and do righteous deeds and establish prayer..."),
            # Prayer decreed upon believers
            (4, 103, "Indeed, prayer has been decreed upon the believers at determined times"),
            # The ones who establish prayer
            (8, 3, "The ones who establish prayer, and from what We have provided them, they spend"),
            # Maintain prayer
            (70, 34, "And those who [carefully] maintain their prayer"),
            # Those who neglect prayer → loss
            (19, 59, "But there came after them successors who neglected prayer and pursued desires"),
        ],
        "expected_hadith": [
            # Islam built on 5 pillars — prayer is one
            ("bukhari", 8, "Islam is based on five principles"),
            ("muslim", 8, "Islam is based on five pillars"),
            # Prayer between man and disbelief
            ("muslim", 82, "Between a man and polytheism and disbelief is the abandonment of prayer"),
        ],
    },

    # ============================================================
    # Q3: "How to perform wudu (ablution)?"
    # ============================================================
    "How to perform wudu (ablution)?": {
        "description": "Wudu — the Quranic verse is 5:6, details are in hadith",
        "expected_quran_standard": [
            # THE verse on wudu — wash faces, forearms, wipe heads, wash feet
            (5, 6, "O you who have believed, when you rise to [perform] prayer, wash your faces..."),
        ],
        "expected_hadith": [
            # Ali performing wudu (detailed steps)
            ("tirmidhi", 48, "I saw Ali performing Wudu. He washed his hands..."),
            # Abdullah bin Zaid showing how the Prophet did wudu
            ("bukhari", 159, "Can you show me how the Messenger of Allah performed ablution?"),
            # Uthman's wudu (complete demonstration)
            ("bukhari", 164, "I saw Uthman bin Affan asking for water to perform ablution..."),
            # Heels dry → Hellfire
            ("nasai", 111, "Woe to the heels from the Hellfire"),
        ],
    },

    # ============================================================
    # Q4: "What does the Quran say about patience in trials?"
    # ============================================================
    "What does the Quran say about patience in trials?": {
        "description": "Patience — verified via alquran.cloud API search for 'be patient' and 'patient'",
        "expected_quran_standard": [
            # If you are patient and fear Allah, their plot will not harm you
            (3, 120, "If you are patient and fear Allah, their plot will not harm you at all"),
            # Patience is most fitting (Prophet Jacob)
            (12, 18, "So patience is most fitting"),
            (12, 83, "So patience is most fitting. Perhaps Allah will bring them to me all together"),
            # Be patient, your patience is through Allah
            (16, 127, "And be patient, [O Muhammad], and your patience is not but through Allah"),
            # Do you think you'll enter Paradise without trial?
            (2, 214, "Or do you think that you will enter Paradise while such has not yet come to you..."),
            # Seek help through Allah and be patient
            (7, 128, "Seek help through Allah and be patient. Indeed, the earth belongs to Allah"),
            # Allah does not allow the reward of those who do good to be lost
            (11, 115, "And be patient, for indeed, Allah does not allow to be lost the reward of those who do good"),
        ],
        "expected_hadith": [
            # Believer is like a fresh tender plant — bends with wind
            ("bukhari", 5648, "The example of a believer is that of a fresh tender plant"),
            # True patience is at the first stroke of calamity
            ("bukhari", 1283, "Verily, patience is at the first stroke of a calamity"),
        ],
    },

    # ============================================================
    # Q5: "What is the ruling on usury (Riba)?"
    # ============================================================
    "What is the ruling on usury (Riba)?": {
        "description": "Usury/Riba — verified via alquran.cloud API search for 'usury' and 'interest'",
        "expected_quran_standard": [
            # Those who consume interest cannot stand
            (2, 275, "Those who consume interest cannot stand except as one stands who is being beaten"),
            # Allah destroys interest and gives increase for charities
            (2, 276, "Allah destroys interest and gives increase for charities"),
            # Give up what remains of interest
            (2, 278, "O you who have believed, fear Allah and give up what remains of interest"),
            # Do not consume usury, doubled and multiplied
            (3, 130, "O you who have believed, do not consume usury, doubled and multiplied"),
            # Taking usury was forbidden
            (4, 161, "And for their taking of usury while they had been forbidden from it"),
        ],
        "expected_hadith": [
            # Curse on the one who consumes Riba
            ("muslim", 3881, "The Messenger of Allah cursed the one who consumes Riba"),
            # A dirham of Riba is worse than 36 acts of zina
            ("ibnmajah", 2274, "A dirham of Riba which a man consumes knowingly is worse than 36 acts of zina"),
        ],
    },
}


# ============================================================
# VERIFIED DB chunk IDs — obtained by running verify_ground_truth.py
# 100% of verses found by matching exact API text against DB text_en.
# All 32 Quran verses + 14 hadiths verified. No missing entries.
# ============================================================

VERIFIED_DB_IDS: dict[str, dict] = {
    "What does the Quran say about charity and zakat?": {
        "quran_chunk_ids": [
            ("quran_9_1295", "9:60 — Zakah expenditures are only for the poor and for t"),
            ("quran_2_50", "2:43 — And establish prayer and give zakah and bow with t"),
            ("quran_2_117", "2:110 — And establish prayer and give zakah, and whatever "),
            ("quran_2_284", "2:277 — Indeed, those who believe and do righteous deeds a"),
            ("quran_2_222", "2:215 — They ask you what they should spend. Say: Whatever"),
            ("quran_70_5400", "70:25 — For the petitioner and the deprived"),
            ("quran_57_5093", "57:18 — Indeed, the men who practice charity and the women"),
            ("quran_19_2281", "19:31 — And He has made me blessed wherever I am and has e"),
            ("quran_2_283", "2:276 — Allah destroys interest and gives increase for cha"),
        ],
        "hadith_chunk_ids": [
            ("hadith_bukhari_1395", "The Prophet sent Mu'adh to Yemen and said... teach"),
            ("hadith_nasai_2538", "Every Muslim must give charity"),
            ("hadith_bukhari_1427", "The best of charity is that given from self-suffic"),
        ],
    },
    "Is prayer obligatory?": {
        "quran_chunk_ids": [
            ("quran_2_10", "2:3 — Who believe in the unseen, establish prayer, and s"),
            ("quran_2_50", "2:43 — And establish prayer and give zakah and bow with t"),
            ("quran_2_117", "2:110 — And establish prayer and give zakah..."),
            ("quran_2_284", "2:277 — Indeed, those who believe and do righteous deeds a"),
            ("quran_4_596", "4:103 — Indeed, prayer has been decreed upon the believers"),
            ("quran_8_1163", "8:3 — The ones who establish prayer, and from what We ha"),
            ("quran_70_5409", "70:34 — And those who [carefully] maintain their prayer"),
            ("quran_19_2309", "19:59 — But there came after them successors who neglected"),
        ],
        "hadith_chunk_ids": [
            ("hadith_bukhari_8", "Islam is based on five principles"),
            ("hadith_muslim_8", "Islam is based on five pillars"),
            ("hadith_muslim_82", "Between a man and polytheism and disbelief is the "),
        ],
    },
    "How to perform wudu (ablution)?": {
        "quran_chunk_ids": [
            ("quran_5_675", "5:6 — O you who have believed, when you rise to [perform"),
        ],
        "hadith_chunk_ids": [
            ("hadith_tirmidhi_48", "I saw Ali performing Wudu. He washed his hands..."),
            ("hadith_bukhari_159", "Can you show me how the Messenger of Allah perform"),
            ("hadith_bukhari_164", "I saw Uthman bin Affan asking for water to perform"),
            ("hadith_nasai_111", "Woe to the heels from the Hellfire"),
        ],
    },
    "What does the Quran say about patience in trials?": {
        "quran_chunk_ids": [
            ("quran_3_413", "3:120 — If you are patient and fear Allah, their plot will"),
            ("quran_12_1614", "12:18 — So patience is most fitting"),
            ("quran_12_1679", "12:83 — So patience is most fitting. Perhaps Allah will br"),
            ("quran_16_2028", "16:127 — And be patient, [O Muhammad], and your patience is"),
            ("quran_2_221", "2:214 — Or do you think that you will enter Paradise while"),
            ("quran_7_1082", "7:128 — Seek help through Allah and be patient. Indeed, th"),
            ("quran_11_1588", "11:115 — And be patient, for indeed, Allah does not allow t"),
        ],
        "hadith_chunk_ids": [
            ("hadith_bukhari_5648", "The example of a believer is that of a fresh tende"),
            ("hadith_bukhari_1283", "Verily, patience is at the first stroke of a calam"),
        ],
    },
    "What is the ruling on usury (Riba)?": {
        "quran_chunk_ids": [
            ("quran_2_282", "2:275 — Those who consume interest cannot stand except as "),
            ("quran_2_283", "2:276 — Allah destroys interest and gives increase for cha"),
            ("quran_2_285", "2:278 — O you who have believed, fear Allah and give up wh"),
            ("quran_3_423", "3:130 — O you who have believed, do not consume usury, dou"),
            ("quran_4_654", "4:161 — And for their taking of usury while they had been "),
        ],
        "hadith_chunk_ids": [
            ("hadith_muslim_3881", "The Messenger of Allah cursed the one who consumes"),
            ("hadith_ibnmajah_2274", "A dirham of Riba which a man consumes knowingly is"),
        ],
    },
}


def get_ground_truth_for_query(query: str) -> dict | None:
    """Get the ground truth for a specific query.

    Args:
        query: The user question string (must match exactly).

    Returns:
        A dict with 'expected_quran_standard' and 'expected_hadith' lists,
        or None if no ground truth exists for this query.
    """
    return GROUND_TRUTH.get(query)


def get_expected_global_ayahs(query: str) -> list[tuple[int, int, str]]:
    """Get expected Quran verses as (surah, global_ayah_num, description) tuples.

    Uses VERIFIED chunk IDs from verify_ground_truth.py — no offset calculation.
    The verified IDs were obtained by searching the DB by text content.

    Args:
        query: The user question string.

    Returns:
        List of (surah_num, global_ayah_num, description) tuples.
    """
    verified = VERIFIED_DB_IDS.get(query, {})
    quran_ids = verified.get("quran_chunk_ids", [])

    result = []
    for chunk_id, desc in quran_ids:
        # Parse surah and global ayah from chunk ID: "quran_9_1295" → (9, 1295)
        parts = chunk_id.split("_")
        if len(parts) == 3 and parts[0] == "quran":
            surah = int(parts[1])
            global_ayah = int(parts[2])
            result.append((surah, global_ayah, desc))
    return result


def get_expected_hadith_ids(query: str) -> list[tuple[str, int, str]]:
    """Get expected hadiths as (collection_slug, hadith_number, description) tuples.

    Uses VERIFIED chunk IDs from verify_ground_truth.py.

    Args:
        query: The user question string.

    Returns:
        List of (collection_slug, hadith_number, description) tuples.
    """
    verified = VERIFIED_DB_IDS.get(query, {})
    hadith_ids = verified.get("hadith_chunk_ids", [])

    result = []
    for chunk_id, desc in hadith_ids:
        # Parse collection and number from chunk ID: "hadith_bukhari_8" → ("bukhari", 8)
        parts = chunk_id.split("_")
        if len(parts) >= 3 and parts[0] == "hadith":
            collection = parts[1]
            number = int(parts[2])
            result.append((collection, number, desc))
    return result
