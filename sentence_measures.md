#### Column names for Sentence measures
Some features were adapted from the popEye R package ([github](https://github.com/sascha2schroeder/popEye))
The if the column depend on a line assignment then a _ALGORITHM_NAME will be at the end of the name.
- subject: Participant ID
- trial_id: Position of trial in analysis
- item: Item ID
- condition: Condition (if applicable)
- sentence_number: Number of sentence in trial
- sentence: Sentence Text
- number_of_words: Number of words in sentence
- skip: Whether the sentence has been skipped
- nrun: Number of times the sentence has been read
- reread: Whether the sentence has been read more than one time
- reg_in: Whether a regression has been made into the sentence
- reg_out: Whether a regression has been made out of the sentence
- total_n_fixations: Number of fixations made on the sentence
- total_dur: Total sentence reading time
- rate: Reading rate (number of words per minute)
- gopast: Sum of all fixations durations from the time the sentence was entered until it was left to the right (regression path duration)
- gopast_sel: Sum of all fixations on the sentence from the time it was entered until it was left to the right (selective go-past time: regression path dur ation minus the time of the regression path)
-  firstrun_skip: Whether sentence has been skipped during first-pass reading
- firstrun_reg_in: Whether a regression has been made into the sentence during first-pass reading
- firstrun_reg_out: Whether a regression has been made out of the sentence during first-pass reading
- firstpass_n_fixations: Number of fixation made during first-pass reading
- firstpass_dur: First-pass reading time
- firstpass_forward_n_fixations: Number of first-pass forward fixations (landing on one of the upcoming words of a sentence)
- firstpass_forward_dur: Duration of forward fixations during first-pass reading
- firstpass_reread_n_fixations: Number of first-pass rereading fixations (landing one of the words of the sentence that have been read previously)
- firstpass_reread_dur: Duration of rereading fixations during first-pass reading
- lookback_n_fixations: Number of fixations made on the sentence after regressing into it from another sentence
- lookback_dur: Duration of lookback fixations on the sentence
- lookfrom_n_fixations: Number of rereading fixations on another sentence initiated from the sentence
- lookfrom_dur: Duration of lookfrom fixations on the sentence

The forward, rereading, look-back, and look-from measures are computed in similar way as in the SR "Getting Reading Measures" tool (https://www.sr-support.com/thread-350.html) which is based on the Eyelink Analysojia software (developed by the Turku Eye Labs).
