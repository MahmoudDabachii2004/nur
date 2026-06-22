# NUR Retrieval Algorithm Audit Report (With BGE-Reranker-v2-M3)

Generated on: 2026-06-21 14:21:39
Evaluation comparing: **Dense-Only** (ChromaDB), **Sparse-Only** (BGE-M3 JSON Lexical), **Hybrid RRF** (Fusing Dense + Sparse), and **Reranked** (BGE-Reranker-v2-M3).

## 1. Query: "ما حكم صلاة الجماعة" (AR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_abudawud_560` | 2.01159 | 6 | 2 | N/A | Hadith \| Collection: Sunan Abi Dawud \| Narrator: AbuSa'id al-Khudri \| Grade: Ungraded \| Chapter: Chapter: The Obligation To Perform The Salat (Pra... |
| #2 | `hadith_nasai_841` | 1.45182 | 8 | N/A | 7 | Hadith \| Collection: Sunan an-Nasa'i \| Grade: Sahih (Darussalam) \| Chapter: Chapter: Mention of Al-Imamah and the congregation \| باب: باب ذكر الام... |
| #3 | `hadith_nasai_840` | 1.40715 | 1 | 3 | 1 | Hadith \| Collection: Sunan an-Nasa'i \| Grade: Sahih (Darussalam) \| Chapter: Chapter: Mention of Al-Imamah and the congregation \| باب: باب ذكر الام... |
| #4 | `hadith_abudawud_559` | 1.32919 | 3 | 6 | 3 | Hadith \| Collection: Sunan Abi Dawud \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: The Obligation To Perform The Salat (Prayers) \| باب: باب الصلا... |
| #5 | `hadith_nasai_839` | 1.26549 | 2 | 1 | 6 | Hadith \| Collection: Sunan an-Nasa'i \| Grade: Sahih (Darussalam) \| Chapter: Chapter: Mention of Al-Imamah and the congregation \| باب: باب ذكر الام... |

## 2. Query: "الربا وأكل أموال الناس بالباطل" (AR on quran)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `quran_4_654` | 2.36355 | 6 | 8 | N/A | Quran \| Surah: An-Nisa (النساء) \| Verset: 654 \| Révélation: Medinan \| الآية السابقة: فبظلم من ٱلذين هادوا حرمنا عليهم طيبت احلت لهم وبصدهم عن سبيل... |
| #2 | `quran_2_195` | -0.70597 | 3 | 5 | N/A | Quran \| Surah: Al-Baqarah (البقرة) \| Verset: 195 \| Révélation: Medinan \| الآية السابقة: احل لكم ليله ٱلصيام ٱلرفث الي نسائكم هن لباس لكم وانتم لبا... |
| #3 | `quran_4_531` | -0.92883 | N/A | 4 | N/A | Quran \| Surah: An-Nisa (النساء) \| Verset: 531 \| Révélation: Medinan \| الآية السابقة: ٱلذين يبخلون ويامرون ٱلناس بٱلبخل ويكتمون ما ءاتيهم ٱلله من ف... |
| #4 | `quran_2_275` | -1.33229 | 10 | 2 | N/A | Quran \| Surah: Al-Baqarah (البقرة) \| Verset: 275 \| Révélation: Medinan \| الآية السابقة: يايها ٱلذين ءامنوا انفقوا من طيبت ما كسبتم ومما اخرجنا لكم... |
| #5 | `quran_2_276` | -1.94679 | 2 | 3 | 1 | Quran \| Surah: Al-Baqarah (البقرة) \| Verset: 276 \| Révélation: Medinan \| الآية السابقة: ٱلشيطن يعدكم ٱلفقر ويامركم بٱلفحشاء وٱلله يعدكم مغفره منه ... |

## 3. Query: "أحاديث عن الصبر عند المصيبة" (AR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_abudawud_3125` | 2.87260 | 2 | 6 | 3 | Hadith \| Collection: Sunan Abi Dawud \| Narrator: Anas \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: Sickness Which Expiate For Sins \| باب: باب ا... |
| #2 | `hadith_tirmidhi_990` | 2.81898 | 1 | 1 | 4 | Hadith \| Collection: Jami` at-Tirmidhi \| Grade: Sahih (Darussalam) \| Chapter: Chapter: What Has Been Related About Reward For The Sick \| باب: باب ... |
| #3 | `hadith_nasai_1874` | 2.39990 | 4 | 4 | 6 | Hadith \| Collection: Sunan an-Nasa'i \| Grade: Sahih (Darussalam) \| Chapter: Chapter: Wishing For Death \| باب: باب تمني الموت ‏ \| متن الحديث: اخبر... |
| #4 | `hadith_bukhari_1260` | 2.21967 | 8 | 8 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Anas \| Grade: Sahih (by consensus) \| Chapter: Chapter: What is said about funerals, and those wh... |
| #5 | `hadith_bukhari_6885` | 2.13801 | 7 | N/A | 2 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Thabit Al-Bunani \| Grade: Sahih (by consensus) \| Chapter: Chapter: “Obey Allah and obey  the Mes... |

## 4. Query: "صلاة الكسوف وكيفيتها" (AR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_muslim_1999` | 2.70929 | 8 | N/A | 4 | Hadith \| Collection: Sahih Muslim \| Grade: Sahih (by consensus) \| Chapter: Chapter: The Eclipse Prayer \| باب: باب صلاه الكسوف ‏ \| متن الحديث: وحد... |
| #2 | `hadith_tirmidhi_560` | 2.64523 | N/A | N/A | 7 | Hadith \| Collection: Jami` at-Tirmidhi \| Grade: Sahih (Darussalam) \| Chapter: Chapter: (What Has Been Related About) Shortening The Prayer During T... |
| #3 | `hadith_bukhari_727` | 2.57303 | 6 | N/A | 3 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Asma' bint Abi Bakr \| Grade: Sahih (by consensus) \| Chapter: Chapter: How the Adhan for Salat (P... |
| #4 | `hadith_muslim_1983` | 2.43924 | 10 | N/A | 5 | Hadith \| Collection: Sahih Muslim \| Grade: Sahih (by consensus) \| Chapter: Chapter: The Eclipse Prayer \| باب: باب صلاه الكسوف ‏ \| متن الحديث: وحد... |
| #5 | `hadith_ibnmajah_999` | 2.34202 | N/A | N/A | 8 | Hadith \| Collection: Sunan Ibn Majah \| Grade: Sahih (Darussalam) \| Chapter: Chapter: The opening of the Prayer \| باب: باب افتتاح الصلاه \| متن الح... |

## 5. Query: "Ruling on congregational prayer" (EN on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_bukhari_634` | 0.07324 | 2 | N/A | 1 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: `Abdullah bin `Umar \| Grade: Sahih (by consensus) \| Chapter: Chapter: How the Adhan for Salat (P... |
| #2 | `hadith_bukhari_2044` | -0.26795 | N/A | N/A | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Abu Huraira \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has come in the Statement of ... |
| #3 | `hadith_bukhari_90` | -0.27881 | N/A | N/A | 8 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Abu Mas`ud Al-Ansari \| Grade: Sahih (by consensus) \| Chapter: Chapter: The superiority of knowle... |
| #4 | `hadith_abudawud_1068` | -1.03670 | N/A | N/A | 10 | Hadith \| Collection: Sunan Abi Dawud \| Narrator: Tariq ibn Shihab \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: The Obligation To Perform The Sal... |
| #5 | `hadith_abudawud_4381` | -3.58335 | 4 | N/A | 2 | Hadith \| Collection: Sunan Abi Dawud \| Narrator: Wa'il ibn Hujr \| Grade: Ungraded \| Chapter: Chapter: Ruling on one who apostatizes \| باب: باب ال... |

## 6. Query: "What does the Quran say about charity and zakat?" (EN on quran)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `quran_58_5117` | 1.03483 | N/A | N/A | 3 | Quran \| Surah: Al-Mujadilah (المجادلة) \| Verset: 5117 \| Révélation: Medinan \| الآية السابقة: يايها ٱلذين ءامنوا اذا نجيتم ٱلرسول فقدموا بين يدي نج... |
| #2 | `quran_9_1310` | 0.88256 | N/A | N/A | N/A | Quran \| Surah: At-Tawbah (التوبة) \| Verset: 1310 \| Révélation: Medinan \| الآية السابقة: يحلفون بٱلله ما قالوا ولقد قالوا كلمه ٱلكفر وكفروا بعد اسل... |
| #3 | `quran_9_1295` | 0.83414 | N/A | 1 | N/A | Quran \| Surah: At-Tawbah (التوبة) \| Verset: 1295 \| Révélation: Medinan \| الآية السابقة: ولو انهم رضوا ما ءاتيهم ٱلله ورسولهۥ وقالوا حسبنا ٱلله سيؤ... |
| #4 | `quran_2_284` | 0.62312 | N/A | N/A | N/A | Quran \| Surah: Al-Baqarah (البقرة) \| Verset: 284 \| Révélation: Medinan \| الآية السابقة: يمحق ٱلله ٱلربوا ويربي ٱلصدقت وٱلله لا يحب كل كفار اثيم \|... |
| #5 | `quran_57_5093` | 0.58846 | 1 | N/A | 2 | Quran \| Surah: Al-Hadid (الحديد) \| Verset: 5093 \| Révélation: Medinan \| الآية السابقة: ٱعلموا ان ٱلله يحي ٱلارض بعد موتها قد بينا لكم ٱلءايت لعلكم... |

## 7. Query: "Ahadith about good character and manners" (EN on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_bukhari_5806` | 3.66427 | 1 | 3 | 1 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Masruq \| Grade: Sahih (by consensus) \| Chapter: Chapter: Al-Birr and As-Sila \| باب: باب قول الل... |
| #2 | `hadith_bukhari_5800` | 3.41997 | 4 | 7 | 4 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Masruq \| Grade: Sahih (by consensus) \| Chapter: Chapter: Al-Birr and As-Sila \| باب: باب قول الل... |
| #3 | `hadith_ibnmajah_3956` | 3.24279 | 5 | 6 | 8 | Hadith \| Collection: Sunan Ibn Majah \| Grade: Da’if (Darussalam) \| Chapter: Chapter: Indifference towards this world \| باب: باب الزهد في الدنيا ‏.... |
| #4 | `hadith_bukhari_3597` | 2.95667 | 7 | N/A | 5 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: `Abdullah bin `Amr \| Grade: Sahih (by consensus) \| Chapter: Chapter: The virtues of the Companio... |
| #5 | `hadith_abudawud_4801` | 2.86805 | 10 | N/A | N/A | Hadith \| Collection: Sunan Abi Dawud \| Narrator: AbudDarda' \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: Regarding forbearance and the character... |

## 8. Query: "Patience and reward in trials" (EN on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_ibnmajah_3768` | -1.36537 | 1 | N/A | 1 | Hadith \| Collection: Sunan Ibn Majah \| Grade: Hasan (Darussalam) \| Chapter: Chapter: Refraining from harming one who says: La Ilaha Illallah \| باب... |
| #2 | `hadith_ibnmajah_1331` | -1.76520 | 3 | 3 | N/A | Hadith \| Collection: Sunan Ibn Majah \| Grade: Hasan (Darussalam) \| Chapter: Chapter: What was narrated concerning visiting the sick \| باب: باب ما ... |
| #3 | `hadith_tirmidhi_989` | -2.33055 | N/A | 2 | N/A | Hadith \| Collection: Jami` at-Tirmidhi \| Grade: Hasan (Darussalam) \| Chapter: Chapter: What Has Been Related About Reward For The Sick \| باب: باب ... |
| #4 | `hadith_tirmidhi_990` | -2.41783 | 8 | 1 | N/A | Hadith \| Collection: Jami` at-Tirmidhi \| Grade: Sahih (Darussalam) \| Chapter: Chapter: What Has Been Related About Reward For The Sick \| باب: باب ... |
| #5 | `hadith_tirmidhi_3648` | -3.48869 | N/A | 8 | N/A | Hadith \| Collection: Jami` at-Tirmidhi \| Grade: Hasan (Darussalam) \| Chapter: Chapter: What Has Been Related About The Virtue OF The Supplication \... |

## 9. Query: "Inheritance rules in Islam" (EN on quran)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `quran_33_3539` | -0.12118 | N/A | N/A | N/A | Quran \| Surah: Al-Ahzab (الأحزاب) \| Verset: 3539 \| Révélation: Medinan \| الآية السابقة: ٱدعوهم لءابائهم هو اقسط عند ٱلله فان لم تعلموا ءاباءهم فاخ... |
| #2 | `quran_4_669` | -0.91229 | 7 | 1 | N/A | Quran \| Surah: An-Nisa (النساء) \| Verset: 669 \| Révélation: Medinan \| الآية السابقة: فاما ٱلذين ءامنوا بٱلله وٱعتصموا بهۦ فسيدخلهم في رحمه منه وفض... |
| #3 | `quran_4_512` | -1.12904 | 8 | N/A | N/A | Quran \| Surah: An-Nisa (النساء) \| Verset: 512 \| Révélation: Medinan \| الآية السابقة: وليست ٱلتوبه للذين يعملون ٱلسيات حتي اذا حضر احدهم ٱلموت قال ... |
| #4 | `quran_8_1235` | -1.36244 | N/A | N/A | 7 | Quran \| Surah: Al-Anfal (الأنفال) \| Verset: 1235 \| Révélation: Medinan \| الآية السابقة: وٱلذين ءامنوا وهاجروا وجهدوا في سبيل ٱلله وٱلذين ءاووا ونص... |
| #5 | `quran_4_504` | -1.42193 | 10 | 3 | N/A | Quran \| Surah: An-Nisa (النساء) \| Verset: 504 \| Révélation: Medinan \| الآية السابقة: ان ٱلذين ياكلون امول ٱليتمي ظلما انما ياكلون في بطونهم نارا و... |

## 10. Query: "Comment faire les ablutions?" (FR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_bukhari_185` | 0.30939 | 2 | 4 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Yahya Al-Mazini \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has been revealed regardi... |
| #2 | `hadith_bukhari_140` | 0.22432 | 3 | 6 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: `Ata' bin Yasar \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has been revealed regardi... |
| #3 | `hadith_abudawud_135` | 0.19820 | 9 | 8 | N/A | Hadith \| Collection: Sunan Abi Dawud \| Narrator: Abdullah ibn Amr ibn al-'As \| Grade: Ungraded \| Chapter: Chapter: Seclusion While Relieving Onese... |
| #4 | `hadith_bukhari_161` | 0.09783 | 8 | 1 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Abu Huraira \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has been revealed regarding a... |
| #5 | `hadith_bukhari_197` | 0.00629 | 1 | 2 | 1 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: `Abdullah bin Zaid \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has been revealed rega... |

## 11. Query: "Quel est le statut de l'usure Riba?" (FR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_bukhari_2093` | -1.24912 | 1 | 1 | 1 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Ibn `Umar \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has come in the Statement of Al... |
| #2 | `hadith_bukhari_2058` | -2.10202 | 10 | N/A | 9 | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Az-Zuhri from Malik bin Aus \| Grade: Sahih (by consensus) \| Chapter: Chapter: What has come in t... |
| #3 | `hadith_bukhari_4335` | -3.18968 | N/A | 8 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: `Aisha \| Grade: Sahih (by consensus) \| Chapter: What has been said about Fãtihat al-Kitab (i.e.,... |
| #4 | `hadith_muslim_3758` | -3.82162 | N/A | N/A | N/A | Hadith \| Collection: Sahih Muslim \| Grade: Sahih (by consensus) \| Chapter: Chapter: The invalidity of Mulamasah and Munabadhah transactions \| باب:... |
| #5 | `hadith_nasai_4485` | -3.89172 | 6 | 2 | N/A | Hadith \| Collection: Sunan an-Nasa'i \| Grade: Sahih (Darussalam) \| Chapter: Chapter: Encouragement to Earn A Living \| باب: باب الحث علي الكسب ‏‏ \... |

## 12. Query: "Le comportement envers les voisins en Islam" (FR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_bukhari_5789` | -3.71438 | N/A | 8 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Abu Huraira \| Grade: Sahih (by consensus) \| Chapter: Chapter: Al-Birr and As-Sila \| باب: باب قو... |
| #2 | `hadith_muslim_6380` | -3.96318 | 5 | 3 | N/A | Hadith \| Collection: Sahih Muslim \| Grade: Sahih (by consensus) \| Chapter: Chapter: Being Dutiful To One's Parents, And Which Of Them Is More Entit... |
| #3 | `hadith_abudawud_4912` | -3.98357 | N/A | N/A | N/A | Hadith \| Collection: Sunan Abi Dawud \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: Regarding forbearance and the character of the Prophet(pbuh) \|... |
| #4 | `hadith_muslim_6366` | -4.37617 | 9 | 5 | N/A | Hadith \| Collection: Sahih Muslim \| Grade: Sahih (by consensus) \| Chapter: Chapter: Being Dutiful To One's Parents, And Which Of Them Is More Entit... |
| #5 | `hadith_muslim_6526` | -4.55122 | N/A | 10 | N/A | Hadith \| Collection: Sahih Muslim \| Grade: Sahih (by consensus) \| Chapter: Chapter: Being Dutiful To One's Parents, And Which Of Them Is More Entit... |

## 13. Query: "L'importance de la recherche de la science" (FR on hadith)

| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| #1 | `hadith_abudawud_3644` | -8.64666 | 9 | 5 | N/A | Hadith \| Collection: Sunan Abi Dawud \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: Regarding the virtue of knowledge \| باب: باب الحث علي طلب العل... |
| #2 | `hadith_abudawud_3665` | -8.82193 | N/A | 10 | N/A | Hadith \| Collection: Sunan Abi Dawud \| Narrator: Abu Hurayrah \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: Regarding the virtue of knowledge \| ... |
| #3 | `hadith_bukhari_85` | -9.27551 | N/A | N/A | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: Abu Huraira \| Grade: Sahih (by consensus) \| Chapter: Chapter: The superiority of knowledge \| با... |
| #4 | `hadith_bukhari_100` | -9.38120 | 1 | 1 | N/A | Hadith \| Collection: Sahih al-Bukhari \| Narrator: `Abdullah bin `Amr bin Al-`As \| Grade: Sahih (by consensus) \| Chapter: Chapter: The superiority ... |
| #5 | `hadith_abudawud_3642` | -10.06456 | 5 | 3 | N/A | Hadith \| Collection: Sunan Abi Dawud \| Narrator: Kathir ibn Qays \| Grade: Sahih (Al-Albani) \| Chapter: Chapter: Regarding the virtue of knowledge ... |

