// A small JSON reader/writer (Chapter 53): enough for safetensors headers, config.json,
// tokenizer.json (11 MB for Qwen3) and our own model-file header. Objects keep insertion order.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmrt {

class Json {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };
  using Array = std::vector<Json>;
  using Object = std::vector<std::pair<std::string, Json>>;

  Json() = default;
  static Json null() { return Json(); }
  static Json boolean(bool b) { Json j; j.type_ = Type::Bool; j.num_ = b ? 1.0 : 0.0; return j; }
  static Json number(double d) { Json j; j.type_ = Type::Number; j.num_ = d; j.is_int_ = false; return j; }
  static Json integer(int64_t i) { Json j; j.type_ = Type::Number; j.num_ = static_cast<double>(i); j.int_ = i; j.is_int_ = true; return j; }
  static Json string(std::string s) { Json j; j.type_ = Type::String; j.str_ = std::move(s); return j; }
  static Json array(Array a = {}) { Json j; j.type_ = Type::Array; j.arr_ = std::make_shared<Array>(std::move(a)); return j; }
  static Json object(Object o = {}) { Json j; j.type_ = Type::Object; j.obj_ = std::make_shared<Object>(std::move(o)); return j; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const { expect(Type::Bool); return num_ != 0.0; }
  double as_double() const { expect(Type::Number); return num_; }
  int64_t as_int() const { expect(Type::Number); return is_int_ ? int_ : static_cast<int64_t>(num_); }
  const std::string& as_string() const { expect(Type::String); return str_; }
  const Array& as_array() const { expect(Type::Array); return *arr_; }
  const Object& as_object() const { expect(Type::Object); return *obj_; }
  Array& as_array() { expect(Type::Array); return *arr_; }
  Object& as_object() { expect(Type::Object); return *obj_; }

  size_t size() const { return is_array() ? arr_->size() : is_object() ? obj_->size() : 0; }
  const Json& operator[](size_t i) const { return as_array().at(i); }
  // Object lookup: `contains` + `at` (throws) or `get` (null if missing).
  bool contains(std::string_view key) const;
  const Json& at(std::string_view key) const;
  const Json& get(std::string_view key) const;
  void set(std::string key, Json value);   // insert or replace, keeps order
  void push(Json value) { as_array().push_back(std::move(value)); }

  // Convenience getters with defaults, for config files.
  double get_double(std::string_view key, double dflt) const { const Json& j = get(key); return j.is_number() ? j.as_double() : dflt; }
  int64_t get_int(std::string_view key, int64_t dflt) const { const Json& j = get(key); return j.is_number() ? j.as_int() : dflt; }
  bool get_bool(std::string_view key, bool dflt) const { const Json& j = get(key); return j.type_ == Type::Bool ? j.as_bool() : dflt; }
  std::string get_string(std::string_view key, const std::string& dflt) const { const Json& j = get(key); return j.is_string() ? j.as_string() : dflt; }

  static Json parse(std::string_view text);
  std::string dump() const;   // compact, deterministic

 private:
  void expect(Type t) const {
    if (type_ != t) throw std::runtime_error("json: wrong type access");
  }
  Type type_ = Type::Null;
  double num_ = 0.0;
  int64_t int_ = 0;
  bool is_int_ = false;
  std::string str_;
  std::shared_ptr<Array> arr_;
  std::shared_ptr<Object> obj_;
};

std::string read_text_file(const std::string& path);

}  // namespace llmrt
