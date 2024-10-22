#### Column names for Word measures
Some features were adapted from the popEye R package ([github](https://github.com/sascha2schroeder/popEye))
The if the column depend on a line assignment then a _ALGORITHM_NAME will be at the end of the name.

- subject: Subject name or ID
- trial_id: Trial ID
- item: Item ID
- condition: Condition (if applicable)
- word_number: Number of word in trial
- word_length: Number of characters in word
- word_xmin: x-coordinate of left side of bounding box
- word_xmax: x-coordinate of right side of bounding box
- word_ymin: y-coordinate of top of bounding box
- word_ymax: y-coordinate of bottom of bounding box
- word_x_center: x-coordinate of center of bounding box
- word_y_center: y-coordinate of center of bounding box
- assigned_line: Line number to which the word belongs
- word: Text of word
- blink_ALGORITHM_NAME: Variable indicating whether there was blink directly before, during, or directly after the word was fixated
- number_of_fixations_ALGORITHM_NAME: Number of fixations on the word during the whole trial
- initial_fixation_duration_ALGORITHM_NAME: Duration of the initial fixation on that word
- first_of_many_duration_ALGORITHM_NAME: Duration of the initial fixation on that word, but only if there was more than one fixation on the word
- total_fixation_duration_ALGORITHM_NAME: Total time the word was read during the trial in ms (total reading time)
- gaze_duration_ALGORITHM_NAME: The sum duration of all fixations inside a word until the word is exited for the first time
- go_past_duration_ALGORITHM_NAME: Go-past time is the sum duration of all fixations from when the interest area is first entered until when it is first exited to the right, including any regressions to the left that occur during that time period 
- second_pass_duration_ALGORITHM_NAME: Second pass duration is the sum duration of all fixations inside an interest area during the second pass over that interest area.
- initial_landing_position_ALGORITHM_NAME: 
- initial_landing_distance_ALGORITHM_NAME: 
- landing_distances_ALGORITHM_NAME: 
- number_of_regressions_in_ALGORITHM_NAME: 
- singlefix_sac_in_ALGORITHM_NAME: Incoming saccade length (in letters) for the first fixation on the word when it was fixated only once during first-pass reading
- firstrun_nfix_ALGORITHM_NAME: Number of fixations made on the word during first-pass reading
- singlefix_land_ALGORITHM_NAME: Landing position (letter) of the first fixation on the word when it was fixated only once during first-pass reading
- firstrun_skip_ALGORITHM_NAME: Variable indicating whether the word was skipped during first-pass reading
- firstfix_cland_ALGORITHM_NAME: Centered landing position of the first fixation on the word (Vitu et al., 2001: landing position - ((wordlength + 1) / 2))
- singlefix_dur_ALGORITHM_NAME: Duration of the first fixation on the word when it was fixated only once during first-pass reading
- firstrun_gopast_sel_ALGORITHM_NAME: Sum of all fixations on the word from the time it was entered until it was left to the right (selective go-past time: go-past time minus the time of the regression path)
- firstfix_land_ALGORITHM_NAME: Landing position (letter) of the first fixation on the word
- skip_ALGORITHM_NAME: Variable indicating whether the word was fixated in the trial
- firstrun_refix_ALGORITHM_NAME: Variable indicating whether the word was refixated during first-pass reading
- firstrun_reg_out_ALGORITHM_NAME: Variable indicating whether there was a regression from the word during first-pass reading
- blink_ALGORITHM_NAME: 
- firstfix_sac_out_ALGORITHM_NAME: Outgoing saccade length (in letters) for the first fixation on the word
- reread_ALGORITHM_NAME: Variable indicating whether the word was reread at least once during the trial
- refix_ALGORITHM_NAME: Variable indicating whether the word has been refixated at least once during a trial
- reg_in_ALGORITHM_NAME: Variable indicating whether there was at least one regression into the word 
- firstrun_dur_ALGORITHM_NAME: Time the word was read during first-pass reading (gaze duration)
- firstfix_sac_in_ALGORITHM_NAME: Incoming saccade length (in letters) for the first fixation on the word
- singlefix_ALGORITHM_NAME: Variable indicating whether the word was fixated only once during first-pass reading
- firstrun_gopast_ALGORITHM_NAME: Sum of all fixations durations from the time the word was entered until it was left to the right (go-past time/regression path duration)
- nrun_ALGORITHM_NAME: Number of times the word was reread within the trial ("reread" means that it was read again after it has been left to the left or right)
- singlefix_cland_ALGORITHM_NAME: Centred landing position of the first fixation on the word when it was fixated only once during first-pass reading
- reg_out_ALGORITHM_NAME: Variable indicating whether there was at least one regression from the word
- firstfix_dur_ALGORITHM_NAME: Duration of the first fixation on the word (first fixation duration)
- firstfix_launch_ALGORITHM_NAME: Launch site distance (incoming saccade length until the space before the word)
- singlefix_sac_out_ALGORITHM_NAME: Outgoing saccade length (in letters) for the first fixation on the word when it was fixated only once during first-pass reading
- firstrun_reg_in_ALGORITHM_NAME: Variable indicating whether there was a regression into the word during first-pass reading
- singlefix_launch_ALGORITHM_NAME: Launch site distance (incoming saccade length until the space before the word) for the first fixation on the word when it was fixated only once during first-pass reading