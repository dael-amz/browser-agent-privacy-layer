const CUE_RULES = Object.freeze([
  ["CVC", /\b(?:cvv2?|cvc2?|security\s+c[o0]de)\b/iu],
  ["PRIVATE_KEY", /\b(?:private\s+key|begin\s+(?:rsa\s+)?private\s+key)\b/iu],
  ["API_KEY", /\b(?:api[\s_-]*key|client\s+secret|access[\s_-]*key)\b/iu],
  ["AUTH_TOKEN", /\b(?:authorization|bearer|auth[\s_-]*token|access[\s_-]*token)\b/iu],
  ["PASSWORD", /\b(?:password|passcode|passphrase)\b/iu],
  ["CARD_NUMBER", /\b(?:card\s+(?:number|no\.?|#)|credit\s+card|debit\s+card|pan)\b/iu],
  ["GOVERNMENT_ID", /\b(?:social\s+security|ssn|passport|driver'?s?\s+licen[cs]e|tax\s+id)\b/iu],
  ["DOB", /\b(?:date\s+of\s+birth|d[.\s/-]*o[.\s/-]*b[.]?)\b/iu],
  ["BANK_ACCOUNT", /\b(?:routing\s+(?:number|no\.?|#)|bank\s+account|account\s+(?:number|no\.?|#)|iban)\b/iu],
  ["SECRET", /\b(?:secret\s+(?:key|value)|recovery\s+(?:code|phrase)|seed\s+phrase)\b/iu],
  ["PHONE", /\b(?:phone|mobile|telephone|tel|call|sms)\b[^\n]{0,28}\+?\d[\d\s().-]{6,}\d(?:\s*(?:ext|x|extension)\s*\d+)?/iu],
  ["API_KEY", /\b(?:sk[-_](?:test|live)[-_][a-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,})\b/u],
  ["AUTH_TOKEN", /\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*/u],
  ["BANK_ACCOUNT", /\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]){11,30}\b/u],
]);

const VALUE_CUE_RULES = Object.freeze([
  ["EMAIL", /^(?:e-?mail)(?:\s+address)?$/iu],
  ["PHONE", /^(?:phone|mobile|telephone|tel)(?:\s+(?:number|no\.?))?$/iu],
  ["NAME", /^(?:(?:full|legal|preferred|first|last|family|given)\s+name|name)$/iu],
  ["ADDRESS", /^(?:(?:home|mailing|billing|shipping|street)\s+address|address(?:\s+line\s*[12])?)$/iu],
  ["CARD_NUMBER", /^(?:(?:credit|debit)\s+card|card)(?:\s+(?:number|no\.?|#))?$/iu],
  ["CVC", /^(?:cvv2?|cvc2?|security\s+c[o0]de)$/iu],
  ["DOB", /^(?:date\s+of\s+birth|birth\s*date|d[.\s/-]*o[.\s/-]*b[.]?)$/iu],
  ["GOVERNMENT_ID", /^(?:social\s+security(?:\s+number)?|ssn|passport(?:\s+number)?|driver'?s?\s+licen[cs]e|tax\s+id)$/iu],
  ["BANK_ACCOUNT", /^(?:routing(?:\s+(?:number|no\.?|#))?|bank\s+account(?:\s+(?:number|no\.?|#))?|account\s+(?:number|no\.?|#))$/iu],
  ["PASSWORD", /^(?:password|passcode|passphrase)$/iu],
  ["API_KEY", /^(?:api[\s_-]*key|client\s+secret|access[\s_-]*key)$/iu],
  ["AUTH_TOKEN", /^(?:authorization|auth[\s_-]*token|access[\s_-]*token)$/iu],
  ["SECRET", /^(?:secret(?:\s+(?:key|value))?|recovery\s+(?:code|phrase)|seed\s+phrase)$/iu],
]);

export function detectSensitiveCues(text) {
  const labels = new Set();
  for (const [label, pattern] of CUE_RULES) {
    if (pattern.test(text)) labels.add(label);
  }
  return [...labels];
}

export function detectSensitiveValueCue(text) {
  if (typeof text !== "string") return null;
  const normalized = text
    .trim()
    .replace(/\s+/gu, " ")
    .replace(/[\s:*]+$/gu, "");
  for (const [label, pattern] of VALUE_CUE_RULES) {
    if (pattern.test(normalized)) return label;
  }
  return null;
}
