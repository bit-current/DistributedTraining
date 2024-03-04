import time
import logging

class StateManager:
    ONBOARDING = "onboarding"
    FILTERING = "filtering"
    TRAINING = "training"

    # Class variables for shared state
    state = ONBOARDING
    onboarding_start_time = time.time()
    onboarding_time_limit = 30
    training_state_threshold = 2

    @classmethod
    def transition_to_filtering(cls):
        cls.state = cls.FILTERING
        logging.info("Transitioned to filtering state.")

    @classmethod
    def transition_to_training(cls):
        cls.state = cls.TRAINING
        logging.info("Transitioned to training state.")

    @classmethod
    def transition_to_onboarding(cls):
        cls.state = cls.ONBOARDING
        cls.onboarding_start_time = time.time()
        logging.info("Transitioned back to onboarding state.")