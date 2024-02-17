losses = []  # To store losses of responders.
    responded = np.zeros(len(uids), dtype=bool)  # To track whether each uid responded, initialized to False.

    for i, response in enumerate(responses[0]):
        if response.dendrite.status_code == 200 and response.loss != []:
            losses.append(response.loss)
            responded[i] = True  # Mark as responded.

    OUTLIER_THRESHOLD = 2 #FIXME

    # Calculate mean and standard deviation of the losses for responders.
    if losses:  # Ensure there are any losses to calculate stats on.
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        # Calculate z-scores based on these stats.
        z_scores = np.abs((losses - mean_loss) / std_loss) if std_loss > 0 else np.zeros_like(losses)
    else:
        mean_loss, std_loss, z_scores = 0, 0, np.array([])

    # Initialize scores with a default value (e.g., 1) for all.
    scores = np.ones(len(uids))
    # Apply a penalty based on the z-score for outliers among responders.
    outliers = np.array([z > OUTLIER_THRESHOLD for z in z_scores])
    scores[responded] = np.where(outliers, 0.3, 1)  # Update scores for responders based on outlier status.

    # Assign a score of 0 to non-responders.
    scores[~responded] = 0.1
    scores = torch.FloatTensor([score for score in scores]).to(self.device)