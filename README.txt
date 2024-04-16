In addition to the behavioral cloning, I also coded and tried:
  - Training with a custom loss function based off of a scoring funcrtion I wrote
  - Training an actor-critic model using a modified version of that aforementioned scoring function
None of the code or formulas in the README are exact. For example, the equation for the loss function in the code includes
some extra terms to avoid dividing by 0 that I didn't include in this README.

Custom Loss Function
The code including the custom loss function and the training loop for that paradigm is hosted in 'rl_pipeline_custom_loss.py'.
It includes the following high level flow:
  - Load a pretrained model
  - Use pretrained model to run rover after arming, collecting all the processed images as you go
  - Disarm the rover
  - Update model while rover is disarmed using the custom loss function
  - Repeat
The custom loss function was the following equation:
      scoring_martix = matrix same size as bw_image with -1s on edges linearly increasing to 1s in the middle
      loss = dot_prod(scoring_matrix, bw_image) * normalized_predicted_throttle
Theoretially, this loss would be high if the track was on the outside edges of the frame and throttle was high.
It would be low if the track was in the middle of the frame and the throttle was high.
This experiment worked after a few training sessions but would break down after many training sessions, because
backpropogation only affected the throttle, teaching the model when to throttle up and throttle down. In the process
of teaching the model when to throttle, it woudl eventually destroy the internal logic that determined the steering
since the steering was not included in the loss function.

Actor-Critic
The code including the actor-critic paradigm is housed in the 'actor_critic' folder. It includes its own data_gen files
since I had to alter the labels for backpropogation to work.
It includes the following flow:
Train critic:
  - define critic_model(input=[image, throttle, steering], output=score)
  - train critic (image_t, image_t+5):
    - throttle, steering, image =  parse_file(...)
    - pred_score = critic_model([image_t, throttle, steering])
    - actual_score = score_fn(image_t, image_t+5, throttle, steering)
    - backpropogate to predict the actual score
  - score_fn(image_t, image_t+5, throttle, steering):
    # image_t = current frame
    # image_t+5 = 5 frames into the future
    - st_matrix = matrix same size as image with -1s on edge and 1s in center
    - throttle_loss = dot_prod(th_matrix, image_t) * throttle  # high if throttling when track is centered
    - st_matrix = matrix same size as image with -1s on edge and 1s in center
    - steering_loss = dot_prod(st_matrix, image_t+5) # high if track is centered 5 frame in the future
    - return 2 - steering_loss - throttle_loss # should never be below 0 since everything is normalized
Train actor:
  - freeze weights of critic model
  - steering, throttle = actor_model(image)
  - loss = critic_model([image, steering, throttle])
  - label always = 0
  - backpropagate, only updating the actor
