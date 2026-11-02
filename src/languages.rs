#[derive(Clone, Copy, Debug)]
pub struct LanguageOption {
    pub code: &'static str,
    pub english_name: &'static str,
    pub zh_name: &'static str,
    pub qwen_asr: bool,
    pub hy_mt2: bool,
}

macro_rules! language {
    ($code:literal, $english:literal, $zh:literal, $qwen:expr, $hy:expr) => {
        LanguageOption {
            code: $code,
            english_name: $english,
            zh_name: $zh,
            qwen_asr: $qwen,
            hy_mt2: $hy,
        }
    };
}

// whisper.cpp's complete language table. qwen_asr marks the fixed-language
// choices accepted by the bundled Qwen3-ASR-1.7B model; hy_mt2 marks the
// languages exposed as translation targets by the bundled Hy-MT2 model.
pub const RECOGNITION_LANGUAGES: &[LanguageOption] = &[
    language!("en", "English", "英语", true, true),
    language!("zh", "Chinese", "中文", true, true),
    language!("de", "German", "德语", true, true),
    language!("es", "Spanish", "西班牙语", true, true),
    language!("ru", "Russian", "俄语", true, true),
    language!("ko", "Korean", "韩语", true, true),
    language!("fr", "French", "法语", true, true),
    language!("ja", "Japanese", "日语", true, true),
    language!("pt", "Portuguese", "葡萄牙语", true, true),
    language!("tr", "Turkish", "土耳其语", true, true),
    language!("pl", "Polish", "波兰语", true, true),
    language!("ca", "Catalan", "加泰罗尼亚语", false, false),
    language!("nl", "Dutch", "荷兰语", true, true),
    language!("ar", "Arabic", "阿拉伯语", true, true),
    language!("sv", "Swedish", "瑞典语", true, false),
    language!("it", "Italian", "意大利语", true, true),
    language!("id", "Indonesian", "印度尼西亚语", true, true),
    language!("hi", "Hindi", "印地语", true, true),
    language!("fi", "Finnish", "芬兰语", true, false),
    language!("vi", "Vietnamese", "越南语", true, true),
    language!("he", "Hebrew", "希伯来语", false, true),
    language!("uk", "Ukrainian", "乌克兰语", false, true),
    language!("el", "Greek", "希腊语", true, false),
    language!("ms", "Malay", "马来语", true, true),
    language!("cs", "Czech", "捷克语", true, true),
    language!("ro", "Romanian", "罗马尼亚语", true, false),
    language!("da", "Danish", "丹麦语", true, false),
    language!("hu", "Hungarian", "匈牙利语", true, false),
    language!("ta", "Tamil", "泰米尔语", false, true),
    language!("no", "Norwegian", "挪威语", false, false),
    language!("th", "Thai", "泰语", true, true),
    language!("ur", "Urdu", "乌尔都语", false, true),
    language!("hr", "Croatian", "克罗地亚语", false, false),
    language!("bg", "Bulgarian", "保加利亚语", false, false),
    language!("lt", "Lithuanian", "立陶宛语", false, false),
    language!("la", "Latin", "拉丁语", false, false),
    language!("mi", "Maori", "毛利语", false, false),
    language!("ml", "Malayalam", "马拉雅拉姆语", false, false),
    language!("cy", "Welsh", "威尔士语", false, false),
    language!("sk", "Slovak", "斯洛伐克语", false, false),
    language!("te", "Telugu", "泰卢固语", false, true),
    language!("fa", "Persian", "波斯语", true, true),
    language!("lv", "Latvian", "拉脱维亚语", false, false),
    language!("bn", "Bengali", "孟加拉语", false, true),
    language!("sr", "Serbian", "塞尔维亚语", false, false),
    language!("az", "Azerbaijani", "阿塞拜疆语", false, false),
    language!("sl", "Slovenian", "斯洛文尼亚语", false, false),
    language!("kn", "Kannada", "卡纳达语", false, false),
    language!("et", "Estonian", "爱沙尼亚语", false, false),
    language!("mk", "Macedonian", "马其顿语", true, false),
    language!("br", "Breton", "布列塔尼语", false, false),
    language!("eu", "Basque", "巴斯克语", false, false),
    language!("is", "Icelandic", "冰岛语", false, false),
    language!("hy", "Armenian", "亚美尼亚语", false, false),
    language!("ne", "Nepali", "尼泊尔语", false, false),
    language!("mn", "Mongolian", "蒙古语", false, true),
    language!("bs", "Bosnian", "波斯尼亚语", false, false),
    language!("kk", "Kazakh", "哈萨克语", false, true),
    language!("sq", "Albanian", "阿尔巴尼亚语", false, false),
    language!("sw", "Swahili", "斯瓦希里语", false, false),
    language!("gl", "Galician", "加利西亚语", false, false),
    language!("mr", "Marathi", "马拉地语", false, true),
    language!("pa", "Punjabi", "旁遮普语", false, false),
    language!("si", "Sinhala", "僧伽罗语", false, false),
    language!("km", "Khmer", "高棉语", false, true),
    language!("sn", "Shona", "绍纳语", false, false),
    language!("yo", "Yoruba", "约鲁巴语", false, false),
    language!("so", "Somali", "索马里语", false, false),
    language!("af", "Afrikaans", "南非荷兰语", false, false),
    language!("oc", "Occitan", "奥克语", false, false),
    language!("ka", "Georgian", "格鲁吉亚语", false, false),
    language!("be", "Belarusian", "白俄罗斯语", false, false),
    language!("tg", "Tajik", "塔吉克语", false, false),
    language!("sd", "Sindhi", "信德语", false, false),
    language!("gu", "Gujarati", "古吉拉特语", false, true),
    language!("am", "Amharic", "阿姆哈拉语", false, false),
    language!("yi", "Yiddish", "意第绪语", false, false),
    language!("lo", "Lao", "老挝语", false, false),
    language!("uz", "Uzbek", "乌兹别克语", false, false),
    language!("fo", "Faroese", "法罗语", false, false),
    language!("ht", "Haitian Creole", "海地克里奥尔语", false, false),
    language!("ps", "Pashto", "普什图语", false, false),
    language!("tk", "Turkmen", "土库曼语", false, false),
    language!("nn", "Nynorsk", "新挪威语", false, false),
    language!("mt", "Maltese", "马耳他语", false, false),
    language!("sa", "Sanskrit", "梵语", false, false),
    language!("lb", "Luxembourgish", "卢森堡语", false, false),
    language!("my", "Myanmar", "缅甸语", false, true),
    language!("bo", "Tibetan", "藏语", false, true),
    language!("tl", "Filipino", "菲律宾语", true, true),
    language!("mg", "Malagasy", "马达加斯加语", false, false),
    language!("as", "Assamese", "阿萨姆语", false, false),
    language!("tt", "Tatar", "鞑靼语", false, false),
    language!("haw", "Hawaiian", "夏威夷语", false, false),
    language!("ln", "Lingala", "林加拉语", false, false),
    language!("ha", "Hausa", "豪萨语", false, false),
    language!("ba", "Bashkir", "巴什基尔语", false, false),
    language!("jw", "Javanese", "爪哇语", false, false),
    language!("su", "Sundanese", "巽他语", false, false),
    language!("yue", "Cantonese", "粤语", true, true),
];

pub const TRADITIONAL_CHINESE: LanguageOption =
    language!("zh-Hant", "Traditional Chinese", "繁体中文", false, true);
pub const UYGHUR: LanguageOption = language!("ug", "Uyghur", "维吾尔语", false, true);

pub fn recognition_language(code: &str) -> Option<&'static LanguageOption> {
    RECOGNITION_LANGUAGES
        .iter()
        .find(|language| language.code == code)
}

pub fn translation_language(code: &str) -> Option<LanguageOption> {
    if code == TRADITIONAL_CHINESE.code {
        Some(TRADITIONAL_CHINESE)
    } else if code == UYGHUR.code {
        Some(UYGHUR)
    } else {
        recognition_language(code)
            .copied()
            .filter(|language| language.hy_mt2)
    }
}

pub fn translation_languages() -> impl Iterator<Item = LanguageOption> {
    RECOGNITION_LANGUAGES
        .iter()
        .copied()
        .filter(|language| language.hy_mt2)
        .chain([TRADITIONAL_CHINESE, UYGHUR])
}

pub fn qwen_asr_supports(code: &str) -> bool {
    code == "auto"
        || recognition_language(code)
            .map(|language| language.qwen_asr)
            .unwrap_or(false)
}

pub fn language_code(value: &str) -> Option<&'static str> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("auto") {
        return Some("auto");
    }
    RECOGNITION_LANGUAGES
        .iter()
        .find(|language| {
            language.code.eq_ignore_ascii_case(value)
                || language.english_name.eq_ignore_ascii_case(value)
        })
        .map(|language| language.code)
        .or_else(|| {
            [TRADITIONAL_CHINESE, UYGHUR]
                .into_iter()
                .find(|language| {
                    language.code.eq_ignore_ascii_case(value)
                        || language.english_name.eq_ignore_ascii_case(value)
                })
                .map(|language| language.code)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_matches_bundled_model_capabilities() {
        assert_eq!(RECOGNITION_LANGUAGES.len(), 100);
        assert_eq!(translation_languages().count(), 38);
        assert_eq!(
            RECOGNITION_LANGUAGES.iter().filter(|x| x.qwen_asr).count(),
            30
        );
    }
}
