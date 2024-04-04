import time
import torch
import random
import math
import copy

class Averager:
    def __init__(self, model, local_dir, repo_id, hf_token=None):
        self.model = model
        self.local_dir = local_dir
        self.repo_id = repo_id
        self.hf_token = hf_token

    def average_gradients(self, scored_gradients):
        total_score = sum(score for _, score in scored_gradients.values())
        averaged_gradients = {name: torch.zeros_like(grad) for name, (grad, _) in scored_gradients.items()[0][1][0].items()}

        if total_score > 0:
            for key, (gradients, score) in scored_gradients.items():
                weight = score / total_score
                for name, grad in gradients.items():
                    averaged_gradients[name] += grad * weight

        return averaged_gradients

    def apply_averaged_gradients(self, averaged_gradients):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in averaged_gradients:
                    param -= averaged_gradients[name]

    def save_model(self):
        self.model.save_pretrained(self.local_dir)

    def push_to_hf_hub(self, commit_message="Pushing model to Hub"):
        if self.hf_token is not None:
            HfFolder.save_token(self.hf_token)
        
        repo = Repository(local_dir=self.local_dir, repo_id=self.repo_id, clone_from=f"https://huggingface.co/{self.repo_id}")
        repo.push_to_hub(commit_message=commit_message)

    def run_periodic_averaging(self, t, scored_gradients):
        while True:
            start_time = time.time()
            averaged_gradients = self.average_gradients(scored_gradients)
            self.apply_averaged_gradients(averaged_gradients)
            self.save_model()
            self.push_to_hf_hub(commit_message="Updated model with new gradients")
            elapsed_time = time.time() - start_time
            time_to_wait = max(0, t - elapsed_time)
            time.sleep(time_to_wait)


##FIXME where does it get its scored_gradients
class RankedAverager(Averager):
    def __init__(self, model, local_dir, repo_id, hf_token=None, population_size=10, generations=5, mutation_rate=0.1):
        super().__init__(model, local_dir, repo_id, hf_token)
        self.population_size = population_size
        self.generations = generations
        self.population_size = population_size
        self.generations = generations
        self.top_k = top_k  # Number of top performing gradients to consider

    def generate_initial_population(self, gradients, scores):
        # Sort gradients based on scores, and take the top_k
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        top_k_gradients = [gradients[i] for i in top_k_indices]
        population = []
        for _ in range(self.population_size):
            individual = self.combine_gradients(top_k_gradients)
            population.append(individual)
        return population

    def combine_gradients(self, top_k_gradients):
        combined_gradient = {}
        for grad_dict in top_k_gradients:
            for name, grad in grad_dict.items():
                if name not in combined_gradient:
                    combined_gradient[name] = grad.clone()  # Use clone to ensure no inplace operation
                else:
                    if random.random() < 0.5:
                        combined_gradient[name] += grad
        return combined_gradient

    def evaluate_individual(self, individual):
        original_state_dict = copy.deepcopy(self.model.state_dict())
        self.apply_averaged_gradients(individual)
        loss, _ = self.model_validator.evaluate_model()
        self.model.load_state_dict(original_state_dict)  # Restore original state
        return loss

    def rank_selection(self, population):
        ranked = sorted(population, key=self.evaluate_individual)
        return ranked

    def average_gradients(self, gradients, scores):
        population = self.generate_initial_population(gradients, scores)

        for generation in range(self.generations):
            ranked_population = self.rank_selection(population)
            new_population = ranked_population[:self.top_k]  # Keep top_k performers

            # Generate new individuals by combining top performers
            while len(new_population) < self.population_size:
                selected = random.sample(new_population, 2)
                new_individual = self.combine_gradients(selected)
                new_population.append(new_individual)

            population = new_population

        # Returning the best gradients after final generation
        best_gradients = self.rank_selection(population)[0]
        return best_gradients