#include "llmrt/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace llmrt {

bool Json::contains(std::string_view key) const {
  for (const auto& kv : as_object()) if (kv.first == key) return true;
  return false;
}

const Json& Json::at(std::string_view key) const {
  for (const auto& kv : as_object()) if (kv.first == key) return kv.second;
  throw std::runtime_error("json: missing key '" + std::string(key) + "'");
}

const Json& Json::get(std::string_view key) const {
  static const Json kNull;
  if (!is_object()) return kNull;
  for (const auto& kv : *obj_) if (kv.first == key) return kv.second;
  return kNull;
}

void Json::set(std::string key, Json value) {
  for (auto& kv : as_object()) if (kv.first == key) { kv.second = std::move(value); return; }
  obj_->emplace_back(std::move(key), std::move(value));
}

namespace {

struct Parser {
  std::string_view s;
  size_t i = 0;

  [[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("json parse error: ") + what + " at byte " + std::to_string(i));
  }
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
  char peek() { ws(); if (i >= s.size()) fail("unexpected end"); return s[i]; }
  void expect(char c) { if (peek() != c) fail("unexpected character"); ++i; }

  static void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) { out.push_back(static_cast<char>(0xC0 | (cp >> 6))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { out.push_back(static_cast<char>(0xE0 | (cp >> 12))); out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    else { out.push_back(static_cast<char>(0xF0 | (cp >> 18))); out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F))); out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
  }
  uint32_t hex4() {
    if (i + 4 > s.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("bad hex digit");
    }
    return v;
  }
  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (c != '\\') { out.push_back(c); continue; }
      if (i >= s.size()) fail("bad escape");
      char e = s[i++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            i += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          put_utf8(out, cp);
          break;
        }
        default: fail("unknown escape");
      }
    }
    return out;
  }
  Json parse_number() {
    size_t start = i;
    bool integral = true;
    if (s[i] == '-') ++i;
    while (i < s.size()) {
      char c = s[i];
      if (c >= '0' && c <= '9') { ++i; continue; }
      if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { integral = false; ++i; continue; }
      break;
    }
    std::string tok(s.substr(start, i - start));
    if (integral && tok.size() < 19) return Json::integer(std::strtoll(tok.c_str(), nullptr, 10));
    return Json::number(std::strtod(tok.c_str(), nullptr));
  }
  Json parse_value() {
    char c = peek();
    if (c == '{') {
      ++i;
      Json obj = Json::object();
      if (peek() == '}') { ++i; return obj; }
      while (true) {
        std::string key = parse_string();
        expect(':');
        obj.as_object().emplace_back(std::move(key), parse_value());
        char d = peek();
        ++i;
        if (d == '}') break;
        if (d != ',') fail("expected , or }");
      }
      return obj;
    }
    if (c == '[') {
      ++i;
      Json arr = Json::array();
      if (peek() == ']') { ++i; return arr; }
      while (true) {
        arr.as_array().push_back(parse_value());
        char d = peek();
        ++i;
        if (d == ']') break;
        if (d != ',') fail("expected , or ]");
      }
      return arr;
    }
    if (c == '"') return Json::string(parse_string());
    if (c == 't' && s.substr(i, 4) == "true") { i += 4; return Json::boolean(true); }
    if (c == 'f' && s.substr(i, 5) == "false") { i += 5; return Json::boolean(false); }
    if (c == 'n' && s.substr(i, 4) == "null") { i += 4; return Json(); }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    fail("unexpected token");
  }
};

void dump_string(const std::string& s, std::string& out) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
        else out.push_back(static_cast<char>(c));
    }
  }
  out.push_back('"');
}

void dump_value(const Json& j, std::string& out) {
  switch (j.type()) {
    case Json::Type::Null: out += "null"; break;
    case Json::Type::Bool: out += j.as_bool() ? "true" : "false"; break;
    case Json::Type::Number: {
      double d = j.as_double();
      if (d == std::floor(d) && std::fabs(d) < 1e15) { out += std::to_string(static_cast<long long>(d)); }
      else { char buf[32]; std::snprintf(buf, sizeof buf, "%.17g", d); out += buf; }
      break;
    }
    case Json::Type::String: dump_string(j.as_string(), out); break;
    case Json::Type::Array: {
      out.push_back('[');
      bool first = true;
      for (const auto& v : j.as_array()) { if (!first) out.push_back(','); first = false; dump_value(v, out); }
      out.push_back(']');
      break;
    }
    case Json::Type::Object: {
      out.push_back('{');
      bool first = true;
      for (const auto& kv : j.as_object()) {
        if (!first) out.push_back(',');
        first = false;
        dump_string(kv.first, out);
        out.push_back(':');
        dump_value(kv.second, out);
      }
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

Json Json::parse(std::string_view text) {
  Parser p{text};
  Json v = p.parse_value();
  p.ws();
  if (p.i != text.size()) p.fail("trailing characters");
  return v;
}

std::string Json::dump() const {
  std::string out;
  dump_value(*this, out);
  return out;
}

std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace llmrt
