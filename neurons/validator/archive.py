                # Initialize the score for the current miner's response.
                score = np.zeros(len(responses))

                # # Use CUDA if available, otherwise use CPU
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load pre-trained model and tokenizer
                # model_name = 'sshleifer/tiny-gpt2'
                # response = ([], "kmfoda/tiny-random-gpt2", 'wikitext', 4, None)

                # load model weights
                for layer, weight in zip(model.parameters(), response.model_weights):
                    # layer = torch.nn.parameter.Parameter(weight)
                    layer = torch.nn.parameter.Parameter(bt.Tensor.deserialize(weight).clone().detach())

                tokenizer = AutoTokenizer.from_pretrained(response.model_name)
                
                # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
                tokenizer.pad_token = tokenizer.eos_token

                # Move the model to the appropriate device
                model.to(device)

                # Load optimized and scheduler
                if response.optimizer_name == "adam":
                    optimizer = torch.optim.AdamW(model.parameters(), lr = response.lr)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr = response.lr)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=response.steps)  

                # Define encoding function
                def encode(examples):
                    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

                # Select the correct datapoints
                dataset_sample = dataset.select(response.dataset_indices)

                # Encode the dataset
                encoded_dataset = dataset_sample.map(encode, batched=True)

                # Create a PyTorch DataLoader
                dataloader = DataLoader(encoded_dataset, batch_size=response.batch_size)

                # if response.gradients == []:
                #     scores[i] = 0
                #     continue
                # else:
                #     for layer, new_grads in zip(model.named_parameters(),response.gradients):
                #         # layer[1].grad = torch.tensor(bt.Tensor.deserialize(new_grads))
                #         layer[1].grad = bt.Tensor.deserialize(new_grads).clone().detach()
                    
                #     # Adjust gradient
                #     optimizer.step()
                #     scheduler.step() 
                #     optimizer.zero_grad()

                # Train data for one epoch
                for step, batch in enumerate(dataloader):
                    
                    # Move batch to device
                    input_ids = torch.stack(batch['input_ids']).to(device)
                    attention_mask = torch.stack(batch['attention_mask']).to(device)
                    labels = torch.stack(batch["input_ids"]).to(device)

                    # Forward pass
                    outputs = model(
                        input_ids = input_ids, 
                        attention_mask = attention_mask,
                        labels = labels
                    )     
                    
                    # Backward pass
                    loss = outputs.loss
                    print(step)
                    print(loss)
                    # synpase.loss = loss
                    loss.backward()

                    # Adjust gradient
                    optimizer.step()
                    scheduler.step() 
                    optimizer.zero_grad()

                    if step == 10:
                        break

                    outputs = model(
                        input_ids = torch.stack(batch["input_ids"]).to(device), 
                        attention_mask = torch.stack(batch["attention_mask"]).to(device),
                        labels = torch.stack(batch["input_ids"]).to(device)
                    )  

                    print("final loss")
                    correct_loss = float(outputs.loss)

                    for swarm_response in swarm_responses:
                        rmse = math.sqrt(np.square(np.subtract([correct_loss],[response.loss])).mean())
                        # Check if the miner has provided the correct response by doubling the dummy input.
                        # If correct, set their score for this round to 1.
                        if rmse < 0.01:
                            score = 1

                        # Update the global score of the miner.
                        # This score contributes to the miner's weight in the network.
                        # A higher weight means that the miner has been consistently responding correctly.
                        scores[i] = alpha * scores[i] + (1 - alpha) * score

                        # Log the results for monitoring purposes.
                        bt.logging.info(f"Score: {score}")
