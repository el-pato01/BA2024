1. Installiere pandas, numpy, matplotlib, os, pytorch, scipy and time
   Pytorch auf Windows mit cpu: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
             oder mit anaconda: conda install pytorch torchvision torchaudio cpuonly -c pytorch
   Der Rest mit pip install, bzw conda install pandas, numpy, matplotlib, os, scipy, time
3. Öffne code/train_vae/MM_with_new_trained_vae.ipynb
4. Passe den Pfad folgender Code-Zeile an:
   torch.save(all_epochs_dict, fr'C:\Users\yanni\OneDrive\Desktop\BachelorArbeit2024\Code\trained_models\vae_MM_getrennt_{num_simulations}_epochs.pth')
5. Lasse code/train_vae/MM_with_new_trained_vae.ipynb laufen
