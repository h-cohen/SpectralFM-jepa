"""python -m spectral_lejepa export-ensemble [--config configs/inference.yaml] [--output DIR] [--force]"""
from spectral_lejepa.models.export import main_ensemble as main

if __name__ == '__main__':
    main()
