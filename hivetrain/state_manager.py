import time
import logging

class StateManager:
    ONBOARDING = "onboarding"
    FILTERING_1 = "filtering_1"
    FILTERING_2 = "filtering_2"
    TRAINING = "training"

    # Class variables for shared state
    state = ONBOARDING
    onboarding_start_time = time.time()
    onboarding_time_limit = 30
    training_state_threshold = 2
    filtering_time_limit = 60  # Add a time limit for the filtering state
    filtering_start_time = None

    @classmethod
    def transition_to_filtering(cls):
        if cls.state == cls.ONBOARDING
            cls.state = cls.FILTERING_1
        elif cls.state = cls.FILTERING_1:
            cls.state = cls.FILTERING_2
        else: 
            logging.info("Base state is not ONBOARDING or FILTERING_1 can't proceed.")   
            return False
        cls.filtering_start_time = time.time()
        logging.info("Transitioned to filtering state.")
        return True
        
    @classmethod
    def transition_to_training(cls):
        cls.state = cls.TRAINING
        logging.info("Transitioned to training state.")

    @classmethod
    def transition_to_onboarding(cls):
        cls.state = cls.ONBOARDING
        cls.onboarding_start_time = time.time()
        logging.info("Transitioned back to onboarding state.")

    @classmethod
    def is_filtering_time_exceeded(cls):
        if cls.filtering_start_time is None:
            return False
        return time.time() - cls.filtering_start_time > cls.filtering_time_limit