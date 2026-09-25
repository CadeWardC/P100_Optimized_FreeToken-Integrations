#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    const std::vector<std::string> k2_regex = {
        R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
    };
    const std::string accented = " e\xCC\x81";
    const std::string joined = " a\xE2\x80\x8C" "b\xE2\x80\x8D" "c";
    const auto k2_actual = unicode_regex_split("I'm" + accented + joined + " 12345!\r\n", k2_regex, false);
    const std::vector<std::string> k2_expected = {"I", "'m", accented, joined, " ", "123", "45", "!\r\n"};
    if (k2_actual != k2_expected) {
        fprintf(stderr, "unexpected K2-Horizon split\n");
        return 1;
    }

    return 0;
}
