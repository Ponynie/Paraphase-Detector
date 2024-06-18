#!/usr/local/bin/python3

from BackTranslation import BackTranslation

trans = BackTranslation()

#print( trans.searchLanguage('English') )
#print( trans.searchLanguage('German') )

original_sentence = 'Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .'

result = trans.translate(original_sentence, src='en', tmp='de', sleeping=1)
backtranslated_sentence = result.result_text

print(original_sentence)
print(backtranslated_sentence)
