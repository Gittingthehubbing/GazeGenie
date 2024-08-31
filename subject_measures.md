#### Column names for Subject level summary statistics
Some features were adapted from the popEye R package ([github](https://github.com/sascha2schroeder/popEye))
The if the column depend on a line assignment then a _ALGORITHM_NAME will be at the end of the name.

- subject: Subject identifier, taken from filename
- ntrial: Number of trials for the subject
- n_question_correct: Total number of correctly answered questions
- blink: Mean number of blinks across trials
- nfix: Mean number of fixations across trials
- skip_ALGORITHM_NAME: Mean proportion of words that have been skipped during first-pass reading across trials
- saccade_length_ALGORITHM_NAME: Mean (forward) saccade length
- refix_ALGORITHM_NAME: Mean proportion of words that have been refixated across trials
- reg_ALGORITHM_NAME: Mean proportion of words which have been regressed into across trials
- mean_fixation_duration: Mean fixation duration
- total_fix_duration: Mean total reading time across trials