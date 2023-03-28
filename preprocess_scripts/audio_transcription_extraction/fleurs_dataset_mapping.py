import os
from collections import OrderedDict
import json

#"" FLEURS Dataset"""

_FLEURS_LANG_TO_ID = OrderedDict([("Afrikaans", "af"), ("Amharic", "am"), ("Arabic", "ar"), ("Armenian", "hy"), ("Assamese", "as"), ("Asturian", "ast"), ("Azerbaijani", "az"), ("Belarusian", "be"), ("Bengali", "bn"), ("Bosnian", "bs"), ("Bulgarian", "bg"), ("Burmese", "my"), ("Catalan", "ca"), ("Cebuano", "ceb"), ("Mandarin Chinese", "cmn_hans"), ("Cantonese Chinese", "yue_hant"), ("Croatian", "hr"), ("Czech", "cs"), ("Danish", "da"), ("Dutch", "nl"), ("English", "en"), ("Estonian", "et"), ("Filipino", "fil"), ("Finnish", "fi"), ("French", "fr"), ("Fula", "ff"), ("Galician", "gl"), ("Ganda", "lg"), ("Georgian", "ka"), ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"), ("Hausa", "ha"), ("Hebrew", "he"), ("Hindi", "hi"), ("Hungarian", "hu"), ("Icelandic", "is"), ("Igbo", "ig"), ("Indonesian", "id"), ("Irish", "ga"), ("Italian", "it"), ("Japanese", "ja"), ("Javanese", "jv"), ("Kabuverdianu", "kea"), ("Kamba", "kam"), ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"), ("Korean", "ko"), ("Kyrgyz", "ky"), ("Lao", "lo"), ("Latvian", "lv"), ("Lingala", "ln"), ("Lithuanian", "lt"), ("Luo", "luo"), ("Luxembourgish", "lb"), ("Macedonian", "mk"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"), ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"), ("Northern-Sotho", "nso"), ("Norwegian", "nb"), ("Nyanja", "ny"), ("Occitan", "oc"), ("Oriya", "or"), ("Oromo", "om"), ("Pashto", "ps"), ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"), ("Russian", "ru"), ("Serbian", "sr"), ("Shona", "sn"), ("Sindhi", "sd"), ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"), ("Sorani-Kurdish", "ckb"), ("Spanish", "es"), ("Swahili", "sw"), ("Swedish", "sv"), ("Tajik", "tg"), ("Tamil", "ta"), ("Telugu", "te"), ("Thai", "th"), ("Turkish", "tr"), ("Ukrainian", "uk"), ("Umbundu", "umb"), ("Urdu", "ur"), ("Uzbek", "uz"), ("Vietnamese", "vi"), ("Welsh", "cy"), ("Wolof", "wo"), ("Xhosa", "xh"), ("Yoruba", "yo"), ("Zulu", "zu")])
_FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}

_FLEURS_LANG = sorted(["af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bn_in", "bs_ba", "ca_es", "ceb_ph", "cmn_hans_cn", "yue_hant_hk", "cs_cz", "cy_gb", "da_dk", "de_de", "el_gr", "en_us", "es_419", "et_ee", "fa_ir", "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es", "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu", "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp", "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in", "ko_kr", "ckb_iq", "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt", "luo_ke", "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my", "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw", "oc_fr", "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br", "ro_ro", "ru_ru", "bg_bg", "sd_in", "sk_sk", "sl_si", "sn_zw", "so_so", "sr_rs", "sv_se", "sw_ke", "ta_in", "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao", "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng", "zu_za"])
_FLEURS_LONG_TO_LANG = {_FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k for k in _FLEURS_LANG}
_FLEURS_LANG_TO_LONG = {v: k for k, v in _FLEURS_LONG_TO_LANG.items()}

_FLEURS_GROUP_TO_LONG = OrderedDict({
    "western_european_we": ["Asturian", "Bosnian", "Catalan", "Croatian", "Danish", "Dutch", "English", "Finnish", "French", "Galician", "German", "Greek", "Hungarian", "Icelandic", "Irish", "Italian", "Kabuverdianu", "Luxembourgish", "Maltese", "Norwegian", "Occitan", "Portuguese", "Spanish", "Swedish", "Welsh"],
    "eastern_european_ee": ["Armenian", "Belarusian", "Bulgarian", "Czech", "Estonian", "Georgian", "Latvian", "Lithuanian", "Macedonian", "Polish", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Ukrainian"],
    "central_asia_middle_north_african_cmn": ["Arabic", "Azerbaijani", "Hebrew", "Kazakh", "Kyrgyz", "Mongolian", "Pashto", "Persian", "Sorani-Kurdish", "Tajik", "Turkish", "Uzbek"],
    "sub_saharan_african_ssa": ["Afrikaans", "Amharic", "Fula", "Ganda", "Hausa", "Igbo", "Kamba", "Lingala", "Luo", "Northern-Sotho", "Nyanja", "Oromo", "Shona", "Somali", "Swahili", "Umbundu", "Wolof", "Xhosa", "Yoruba", "Zulu"],
    "south_asian_sa": ["Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Oriya", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"],
    "south_east_asian_sea": ["Burmese", "Cebuano", "Filipino", "Indonesian", "Javanese", "Khmer", "Lao", "Malay", "Maori", "Thai", "Vietnamese"],
    "chinese_japanase_korean_cjk": ["Mandarin Chinese", "Cantonese Chinese", "Japanese", "Korean"],
})
_FLEURS_LONG_TO_GROUP = {a: k for k, v in _FLEURS_GROUP_TO_LONG.items() for a in v}
_FLEURS_LANG_TO_GROUP = {_FLEURS_LONG_TO_LANG[k]: v for k, v in _FLEURS_LONG_TO_GROUP.items()}

#print(dict(_FLEURS_LANG_TO_ID ))
_FLEURS_ID_TO_LANG= {v: k for k, v in _FLEURS_LANG_TO_ID.items()}

#save the FLEURS lang to id mapping
with open("../../data/SAIM_data/language_metadata/FLEURS_lang_to_id.json", "w") as f:
    json.dump(_FLEURS_LANG_TO_ID, f, indent=4)

#ID to lang
with open("../../data/SAIM_data/language_metadata/FLEURS_id_to_lang.json", "w") as f:
    json.dump(_FLEURS_ID_TO_LANG, f, indent=4)
