from textstat.textstat import textstat as ts


article_txt = ...

# === READABILITY ===
flesch_reading_score  = ts.flesch_reading_ease(article_txt)
smog_score            = ts.smog_index(article_txt)
flesch_kincaid_score  = ts.flesch_kincaid_grade(test_data)
coleman_liau_score    = ts.coleman_liau_index(test_data)
ari_score             = ts.automated_readability_index(test_data)
dale_chall_score      = ts.dale_chall_readability_score(test_data)
linsear_write_score   = ts.linsear_write_formula(test_data)
gunning_fog_score     = ts.gunning_fog(test_data)
readability_consensus = ts.text_standard(test_data)

print(("Flesch Reading Ease:          {}\n" +
       "Flesch Kincaid Grade:         {}\n" +
       "SMOG Index:                   {}\n" +
       "Automated Readability Index:  {}\n" +
       "Dale Chall Readability Score: {}\n" +
       "Linsear Write Score:          {}\n" +
       "Gunning Fog:                  {}\n" +
       "-" * 79                             +
       "Readability Consensus:        {}").format(
          flesch_reading_score, flesch_kincaid_grade, smog_score,
          ari_score, dale_chall_score, linsear_write_score,
          gunning_fog_score, readability_consensus
          ))


