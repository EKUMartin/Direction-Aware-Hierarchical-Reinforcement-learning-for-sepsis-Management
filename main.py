import torch
# main.py
if __name__ == "__main__":
    # Getting Whole Data


    # Sampling 1000 for pretraining
    sampler = Sample(data=df_query, iterations=1000, threshold=0.05, sample_size=1000)
    df_sampled = sampler.main()
    
    # HMM training
    features_col = ['pao2', 'fio2', 'platelets', 'bilirubin', 'creatinine', 'lactate', 'gcs']
    hmm_module = SepsisHMM(n_components=4)
    hmm_module.train(df_sampled, features_col)
    hmm_module.save_model()
    
    # Label for Encoder
    df_sampled['hmm_state'] = hmm_module.predict(df_sampled, features_col)

    # scaling before going into the encoder
    X_scaled = hmm_module.scaler.transform(df_sampled[features_col].values)
    hmm_states = df_sampled['hmm_state'].values
    tensor_X = torch.FloatTensor(X_scaled)
    tensor_state = torch.LongTensor(hmm_states)
    dataset = TensorDataset(tensor_X, tensor_state)
    train_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    encoder_module = SepsisEncoder(
        input_dim=len(features_col), 
        latent_dim=8, 
        device='cuda'
    )
    
    encoder_module.train(train_dataloader, epochs=20)
    encoder_module.save_model(path='sde_encoder_dict.pth')