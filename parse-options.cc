// util/parse-options.cc

// Copyright 2009-2011  Karel Vesely;  Microsoft Corporation;
//                      Saarland University (Author: Arnab Ghoshal);
// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey);
//                      Frantisek Skala;  Arnab Ghoshal
// Copyright 2013       Tanel Alumae
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sstream>
#include "parse-options.h"

void ParseOptions::Register(const std::string &name,
                            bool *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}

void ParseOptions::Register(const std::string &name,
                            int32 *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}

void ParseOptions::Register(const std::string &name,
                            uint32 *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}

void ParseOptions::Register(const std::string &name,
                            float *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}

void ParseOptions::Register(const std::string &name,
                            double *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}

void ParseOptions::Register(const std::string &name,
                            std::string *ptr, const std::string &doc) {
  RegisterTmpl(name, ptr, doc);
}

// old-style, used for registering application-specific parameters
template<typename T>
void ParseOptions::RegisterTmpl(const std::string &name, T *ptr,
                                const std::string &doc) {
  if (other_parser_ == NULL) {
    this->RegisterCommon(name, ptr, doc, false);
  } else {
    assert(prefix_ != "" &&
                 "Cannot use empty prefix when registering with prefix.");
    std::string new_name = prefix_ + '.' + name;  // --name becomes --prefix.name
    other_parser_->RegisterCommon(new_name, ptr, doc, false);
  }
}

// does the common part of the job of registering a parameter
template<typename T>
void ParseOptions::RegisterCommon(const std::string &name, T *ptr,
                                  const std::string &doc, bool is_standard) {
  assert(ptr != NULL);
  std::string idx = name;
  NormalizeArgName(&idx);
  if (doc_map_.find(idx) != doc_map_.end())
    printf("Registering option twice, ignoring second time: %s\n",name.c_str());
  this->RegisterSpecific(name, idx, ptr, doc, is_standard);
}

// used to register standard parameters (those that are present in all of the
// applications)
template<typename T>
void ParseOptions::RegisterStandard(const std::string &name, T *ptr,
                            const std::string &doc) {
  this->RegisterCommon(name, ptr, doc, true);
}

void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    bool *b,
                                    const std::string &doc,
                                    bool is_standard) {
  bool_map_[idx] = b;
  doc_map_[idx] = DocInfo(name, doc + " (bool, default = "
                          + ((*b)? "true)" : "false)"), is_standard);
}

void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    int32 *i,
                                    const std::string &doc,
                                    bool is_standard) {
  int_map_[idx] = i;
  std::ostringstream ss;
  ss << doc << " (int, default = " << *i << ")";
  doc_map_[idx] = DocInfo(name, ss.str(), is_standard);
}

void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    uint32 *u,
                                    const std::string &doc,
                                    bool is_standard) {
  uint_map_[idx] = u;
  std::ostringstream ss;
  ss << doc << " (uint, default = " << *u << ")";
  doc_map_[idx] = DocInfo(name, ss.str(), is_standard);
}

void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    float *f,
                                    const std::string &doc,
                                    bool is_standard) {
  float_map_[idx] = f;
  std::ostringstream ss;
  ss << doc << " (float, default = " << *f << ")";
  doc_map_[idx] = DocInfo(name, ss.str(), is_standard);
}

void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    double *f,
                                    const std::string &doc,
                                    bool is_standard) {
  double_map_[idx] = f;
  std::ostringstream ss;
  ss << doc << " (double, default = " << *f << ")";
  doc_map_[idx] = DocInfo(name, ss.str(), is_standard);
}

void ParseOptions::RegisterSpecific(const std::string &name,
                                    const std::string &idx,
                                    std::string *s,
                                    const std::string &doc,
                                    bool is_standard) {
  string_map_[idx] = s;
  doc_map_[idx] = DocInfo(name, doc + " (string, default = \"" + *s + "\")",
                          is_standard);
}
void ParseOptions::DisableOption(const std::string &name) {
  if (argv_ != NULL){
    fprintf(stderr,"DisableOption must not be called after calling Read().\n");
    exit(0);
  }
  if (doc_map_.erase(name) == 0){
    fprintf( stderr,"Option %s was not registered so cannot be disabled: \n",name.c_str());
    exit(0);
  }
  bool_map_.erase(name);
  int_map_.erase(name);
  uint_map_.erase(name);
  float_map_.erase(name);
  double_map_.erase(name);
  string_map_.erase(name);
}


int ParseOptions::NumArgs() const {
  return positional_args_.size();
}

std::string ParseOptions::GetArg(int i) const {
  if (i < 1 || i > static_cast<int>(positional_args_.size())){
    fprintf(stderr,"ParseOptions::GetArg, invalid index %d\n",i);  // code error
    exit(0);
  }
  // so use KALDI_ERR
  return positional_args_[i - 1];
}

enum ShellType { kBash = 0 };  // We currently do not support any
// other options.

static ShellType kShellType = kBash;  // This can be changed in the
// code if it ever does need to be changed (as it's unlikely that one
// compilation of this tool-set would use both shells).

static bool MustBeQuoted(const std::string &str, ShellType st) {
  // returns true if we need to escape it before putting it into
  // a shell (mainly thinking of bash shell, but should work for others)
  // This is for the convenience of the user so command-lines that are
  // printed out by ParseOptions::Read (with --print-args=true) are
  // paste-able into the shell and will run.
  // If you use a different type of shell, it might be necessary to
  // change this function.
  // But it's mostly a cosmetic issue as it basically affects how
  // the program echoes its command-line arguments to the screen.

  assert(st == kBash);  // Nothing else supported right now.
  const char *c = str.c_str();
  if (*c == '\0') return true;  // Must quote empty string
  else {
    const char *ok_chars[2];
    ok_chars[kBash] = "[]~#^_-+=:.,/";  // these seem not to be interpreted as long
    // as there are no other "bad" characters involved (e.g. "," would be interpreted
    // as part of something like a{b,c}, but not on its own.

    assert(!strchr(ok_chars[kBash], ' ')); // Just want to make sure that
    // a space character doesn't get automatically inserted here via an automated
    // style-checking script, like it did before.
    
    for (; *c != '\0'; c++) {
      if ( ! isalnum(*c) ) {
        // For non-alphanumeric characters we have a list of
        // characters which are OK.  All others are forbidden
        // (this is easier since the shell interprets most non-alphanumeric
        // characters).
        const char *d;
        for (d = ok_chars[st]; *d != '\0'; d++) if (*c == *d) break;
        if (*d == '\0') return true;  // Was not alphanumeric, or
        // one of the "ok_chars".  So must be escaped
      }
    }
    return false;  // The string was OK: no quoting or escaping.
  }
}

// returns a quoted and escaped version of "str"
// which has previously been determined to need escaping.
// Our aim is to print out the command line in such a way that if it's
// pasted into a shell of ShellType "st" (only bash for now), it
// will get passed to the program in the same way.  For now
// we use the following rules:
//  In the normal case, we quote with single-quote "'", and to escape
// a single-quote we use the string: '\'' (interpreted as closing the
// single-quote, putting an escaped single-quote from the shell, and
// then reopening the single quote).
//  If the string contains single-quotes that would need escaping this
// way, and we determine that the string could be safely double-quoted
// without requiring any escaping, then we double-quote the string.
// This is the case if the characters "`$\ do not appear in the string.
// e.g. see http://www.redhat.com/mirrors/LDP/LDP/abs/html/quotingvar.html

static std::string QuoteAndEscape(const std::string &str, ShellType st) {
  char quote_char;
  const char *escape_str;  // the sequence of characters we insert
  // when we encounter a quote character.

  if (st == kBash) {
    const char *c_str = str.c_str();
    if (strchr(c_str, '\'') && !strpbrk(c_str, "\"`$\\")) {
      quote_char = '"';
      escape_str = "\\\"";  // should never be accessed.
    } else {
      quote_char = '\'';
      escape_str = "'\\''";  // e.g. echo 'a'\''b' returns a'b
    }
  } else {
    fprintf(stderr,"Invalid shell type.\n");
    exit(0);
  }

  char buf[2];
  buf[1] = '\0';

  buf[0] = quote_char;
  std::string ans = buf;
  const char *c = str.c_str();
  for (;*c != '\0'; c++) {
    if (*c == quote_char) {
      ans += escape_str;
    } else {
      buf[0] = *c;
      ans += buf;
    }
  }
  buf[0] = quote_char;
  ans += buf;
  return ans;
}

// static function
std::string ParseOptions::Escape(const std::string &str) {
  if (!MustBeQuoted(str, kShellType)) return str;
  else return QuoteAndEscape(str, kShellType);
}




int ParseOptions::Read(int argc, const char *const argv[]) {
  argc_ = argc;
  argv_ = argv;
  std::string key, value;
  int i;
  if (argc > 0) {
    // set global "const char*" g_program_name
    // name of the program (followed by ':')
    // so it can print it out in error messages;
    // it's useful because often the stderr of different programs will
    // be mixed together in the same log file.
#ifdef _MSC_VER
    const char *c = strrchr(argv[0], '\\');
#else
    const char *c = strrchr(argv[0], '/');
#endif
    if (c == NULL) c = argv[0];
    else c++;
    program_name = new char[strlen(c)+2];
    strcpy(program_name, c);
    strcat(program_name, ":");
//    if (g_program_name != NULL)
//      delete [] g_program_name;
//    g_program_name = program_name;
  }
  // first pass: look for config parameter, look for priority
  for (i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], "--", 2) == 0) {
      if (std::strcmp(argv[i], "--") == 0) {
        // a lone "--" marks the end of named options
        break;
      }
      bool has_equal_sign;
      SplitLongArg(argv[i], &key, &value, &has_equal_sign);
      NormalizeArgName(&key);
      Trim(&value);
      if (key.compare("config") == 0) {
        ReadConfigFile(value);
      }
      if (key.compare("help") == 0) {
        PrintUsage();
        exit(0);
      }
    }
  }
  bool double_dash_seen = false;
  // second pass: add the command line options
  for (i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], "--", 2) == 0) {
      if (std::strcmp(argv[i], "--") == 0) {
        // A lone "--" marks the end of named options.
        // Skip that option and break the processing of named options
        i += 1;
        double_dash_seen = true;
        break;
      }
      bool has_equal_sign;
      SplitLongArg(argv[i], &key, &value, &has_equal_sign);
      NormalizeArgName(&key);
      Trim(&value);
      if (!SetOption(key, value, has_equal_sign)) {
        PrintUsage(true);
        fprintf(stderr,"Invalid option %s\n",argv[i]);
        exit(0);
      }
    } else {
      break;
    }
  }
  
  // process remaining arguments as positional
  for (; i < argc; i++) {
    if ((std::strcmp(argv[i], "--") == 0) && !double_dash_seen) {
      double_dash_seen = true;
    } else {
      positional_args_.push_back(std::string(argv[i]));
    }
  }

  if (print_args_) {  // if the user did not suppress this with --print-args = false....
    std::ostringstream strm;
    for (int j = 0; j < argc; j++)
      strm << Escape(argv[j]) << " ";
    strm << '\n';
    std::cerr << strm.str() << std::flush;
  }
  return i;
}


void ParseOptions::PrintUsage(bool print_command_line) {
  printf("in usage\n");
  std::cerr << '\n' << usage_ << '\n';
  DocMapType::iterator it;
  // first we print application-specific options
  bool app_specific_header_printed = false;
  for (it = doc_map_.begin(); it != doc_map_.end(); ++it) {
    if (it->second.is_standard_ == false) {  // we have application-specific option
      if (app_specific_header_printed == false) {  // header was not yet printed
        std::cerr << "Options:" << '\n';
        app_specific_header_printed = true;
      }
      std::cerr << "  --" << std::setw(25) << std::left << it->second.name_
          << " : " << it->second.use_msg_ << '\n';
    }
  }
  if (app_specific_header_printed == true) {
    std::cerr << '\n';
  }

  // then the standard options
  std::cerr << "Standard options:" << '\n';
  for (it = doc_map_.begin(); it != doc_map_.end(); ++it) {
    if (it->second.is_standard_ == true) {  // we have standard option
      std::cerr << "  --" << std::setw(25) << std::left << it->second.name_
          << " : " << it->second.use_msg_ << '\n';
    }
  }
  std::cerr << '\n';
  if (print_command_line) {
    std::ostringstream strm;
    strm << "Command line was: ";
    for (int j = 0; j < argc_; j++)
      strm << Escape(argv_[j]) << " ";
    strm << '\n';
    std::cerr << strm.str() << std::flush;
  }
}

void ParseOptions::PrintConfig(std::ostream &os) {
  os << '\n' << "[[ Configuration of UI-Registered options ]]"
      << '\n';
  std::string key;
  DocMapType::iterator it;
  for (it = doc_map_.begin(); it != doc_map_.end(); ++it) {
    key = it->first;
    os << it->second.name_ << " = ";
    if (bool_map_.end() != bool_map_.find(key)) {
      os << (*bool_map_[key] ? "true" : "false");
    } else if (int_map_.end() != int_map_.find(key)) {
      os << (*int_map_[key]);
    } else if (uint_map_.end() != uint_map_.find(key)) {
      os << (*uint_map_[key]);
    } else if (float_map_.end() != float_map_.find(key)) {
      os << (*float_map_[key]);
    } else if (double_map_.end() != double_map_.find(key)) {
      os << (*double_map_[key]);
    } else if (string_map_.end() != string_map_.find(key)) {
      os << "'" << *string_map_[key] << "'";
    } else {
      fprintf(stderr,"PrintConfig: unrecognized option %s\n",key.c_str());
      exit(0);
    }
    os << '\n';
  }
  os << '\n';
}


void ParseOptions::ReadConfigFile(const std::string &filename) {
  std::ifstream is(filename.c_str(), std::ifstream::in);
  if (!is.good()) {
    fprintf(stderr,"Cannot open config file: %s\n",filename.c_str());
    exit(0);
  }

  std::string line, key, value;
  int32 line_number = 0;
  while (std::getline(is, line)) {
    line_number++;
    // trim out the comments
    size_t pos;
    if ((pos = line.find_first_of('#')) != std::string::npos) {
      line.erase(pos);
    }
    // skip empty lines
    Trim(&line);
    if (line.length() == 0) continue;

    if (line.substr(0, 2) != "--") {
      fprintf(stderr,"Reading config file %s\n",filename.c_str());
      exit(0);
//                << ": line " << line_number << " does not look like a line "
//                << "from a Kaldi command-line program's config file: should "
//                << "be of the form --x=y.  Note: config files intended to "
//                << "be sourced by shell scripts lack the '--'.";
    }
    
    // parse option
    bool has_equal_sign;
    SplitLongArg(line, &key, &value, &has_equal_sign);
    NormalizeArgName(&key);
    Trim(&value);
    if (!SetOption(key, value, has_equal_sign)) {
      PrintUsage(true);
      exit(0);
    }
  }
}



void ParseOptions::SplitLongArg(std::string in,
                                std::string *key,
                                std::string *value,
                                bool *has_equal_sign) {
  assert(in.substr(0, 2) == "--");  // precondition.
  size_t pos = in.find_first_of('=', 0);
  if (pos == std::string::npos) {  // we allow --option for bools
    // defaults to empty.  We handle this differently in different cases.
    *key = in.substr(2, in.size()-2);  // 2 because starts with --.
    *value = "";
    *has_equal_sign = false;
  } else if (pos == 2) {  // we also don't allow empty keys: --=value
    PrintUsage(true);
    exit(0);
  } else {  // normal case: --option=value
    *key = in.substr(2, pos-2);  // 2 because starts with --.
    *value = in.substr(pos + 1);
    *has_equal_sign = true;
  }
}

void ParseOptions::Trim(std::string* str) {
  const char *white_chars = " \t\n\r\f\v";

  std::string::size_type pos = str->find_last_not_of(white_chars);
  if (pos != std::string::npos)  {
    str->erase(pos + 1);
    pos = str->find_first_not_of(white_chars);
    if (pos != std::string::npos) str->erase(0, pos);
  } else {
    str->erase(str->begin(), str->end());
  }
}

void ParseOptions::NormalizeArgName(std::string *str) {
  std::string out;
  std::string::iterator it;

  for (it = str->begin(); it != str->end(); ++it) {
    if (*it == '_') out += '-';  // convert _ to -
    else out += std::tolower(*it);
  }
  *str = out;

  assert(str->length() > 0);
}




bool ParseOptions::SetOption(const std::string &key,
                             const std::string &value,
                             bool has_equal_sign) {
  if (bool_map_.end() != bool_map_.find(key)) {
    if (has_equal_sign && value == ""){
      fprintf(stderr,"Invalid option --%s=\n",key.c_str());
      exit(0);
    }
    *(bool_map_[key]) = ToBool(value);
  } else if (int_map_.end() != int_map_.find(key)) {
    *(int_map_[key]) = ToInt(value);
  } else if (uint_map_.end() != uint_map_.find(key)) {
    *(uint_map_[key]) = ToUInt(value);
  } else if (float_map_.end() != float_map_.find(key)) {
    *(float_map_[key]) = ToFloat(value);
  } else if (double_map_.end() != double_map_.find(key)) {
    *(double_map_[key]) = ToDouble(value);
  } else if (string_map_.end() != string_map_.find(key)) {
    if (!has_equal_sign){
      fprintf(stderr,"Invalid option --%s\n",key.c_str());
      exit(0);
    }
    *(string_map_[key]) = value;
  } else {
    return false;
  }
  return true;
}



bool ParseOptions::ToBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  // allow "" as a valid option for "true", so that --x is the same as --x=true
  if ((str.compare("true") == 0) || (str.compare("t") == 0)
      || (str.compare("1") == 0) || (str.compare("") == 0)) {
    return true;
  }
  if ((str.compare("false") == 0) || (str.compare("f") == 0)
      || (str.compare("0") == 0)) {
    return false;
  }
  // if it is neither true nor false:
  PrintUsage(true);
  exit(0);
  return false;  // never reached
}


int32 ParseOptions::ToInt(std::string str) {
  char *end_pos;
  // strtol is cheaper than stringstream...
  // strtol accepts decimal 438143, hexa 0x1f2d3 and octal 067123
  int32 ret = std::strtol(str.c_str(), &end_pos, 0);
  if (str.c_str() == end_pos) {
    PrintUsage(true);
    exit(0);
  }
  return ret;
}

uint32 ParseOptions::ToUInt(std::string str) {
  char *end_pos;
  // strtol is cheaper than stringstream...
  // strtol accepts decimal 438143, hexa 0x1f2d3 and octal 067123
  uint32 ret = std::strtoul(str.c_str(), &end_pos, 0);
  if (str.c_str() == end_pos) {
    PrintUsage(true);
    exit(0);
  }
  return ret;
}


float ParseOptions::ToFloat(std::string str) {
  char *end_pos;
  // strtod is cheaper than stringstream...
  float ret = std::strtod(str.c_str(), &end_pos);
  if (str.c_str() == end_pos) {
    PrintUsage(true);
    exit(0);
  }
  return ret;
}

double ParseOptions::ToDouble(std::string str) {
  char *end_pos;
  // strtod is cheaper than stringstream...
  double ret = std::strtod(str.c_str(), &end_pos);
  if (str.c_str() == end_pos) {
    PrintUsage(true);
    exit(0);
  }
  return ret;
}

// instantiate templates
template void ParseOptions::RegisterTmpl(const std::string &name, bool *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterTmpl(const std::string &name, int32 *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterTmpl(const std::string &name, uint32 *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterTmpl(const std::string &name, float *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterTmpl(const std::string &name, double *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterTmpl(const std::string &name, std::string *ptr,
                            const std::string &doc);

template void ParseOptions::RegisterStandard(const std::string &name,
                            bool *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterStandard(const std::string &name,
                            int32 *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterStandard(const std::string &name,
                            uint32 *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterStandard(const std::string &name,
                            float *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterStandard(const std::string &name,
                            double *ptr,
                            const std::string &doc);
template void ParseOptions::RegisterStandard(const std::string &name,
                            std::string *ptr,
                            const std::string &doc);

template void ParseOptions::RegisterCommon(const std::string &name,
                            bool *ptr,
                            const std::string &doc, bool is_standard);
template void ParseOptions::RegisterCommon(const std::string &name,
                            int32 *ptr,
                            const std::string &doc, bool is_standard);
template void ParseOptions::RegisterCommon(const std::string &name,
                            uint32 *ptr,
                            const std::string &doc, bool is_standard);
template void ParseOptions::RegisterCommon(const std::string &name,
                            float *ptr,
                            const std::string &doc, bool is_standard);
template void ParseOptions::RegisterCommon(const std::string &name,
                            double *ptr,
                            const std::string &doc, bool is_standard);
template void ParseOptions::RegisterCommon(const std::string &name,
                            std::string *ptr,
                            const std::string &doc, bool is_standard);
