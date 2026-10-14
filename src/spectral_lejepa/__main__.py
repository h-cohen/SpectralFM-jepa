"""python -m spectral_lejepa <command> [args]: one entry point for every pipeline step."""
import importlib
import sys

COMMANDS = {
    'pretrain': 'spectral_lejepa.training.pretrain',
    'gradient-audit': 'spectral_lejepa.training.audit',
    'export': 'spectral_lejepa.models.export',
    'export-ensemble': 'spectral_lejepa.models.export_ensemble',
    'embed': 'spectral_lejepa.inference',
    'pack': 'spectral_lejepa.data.pack_spectra',
    'pack-pickles': 'spectral_lejepa.data.pack_pickles',
    'noise-calibration': 'spectral_lejepa.data.noise_calibration',
    'evaluate': 'spectral_lejepa.evaluation.run',
    'gate': 'spectral_lejepa.evaluation.gate',
    'ensemble': 'spectral_lejepa.evaluation.ensemble',
    'permutation-audit': 'spectral_lejepa.evaluation.permutation',
    'perturbation': 'spectral_lejepa.evaluation.perturbation',
    'robustness': 'spectral_lejepa.evaluation.perturbation_eval',
    'report': 'spectral_lejepa.reporting.results',
    'training-report': 'spectral_lejepa.reporting.training_report',
    'training-report-html': 'spectral_lejepa.reporting.training_report_html',
    'embedding-figures': 'spectral_lejepa.reporting.embedding_figures',
}

# Commands whose main() parses no options, with the positional words each accepts. Help is answered
# here so that `--help` never starts the job.
NO_OPTIONS = {'training-report': ('fetch',), 'training-report-html': (), 'noise-calibration': (), 'perturbation': (),
              'embedding-figures': ()}
USAGE = 'usage: python -m spectral_lejepa {' + ','.join(COMMANDS) + '} [args]\n'
HELP = ('-h', '--help')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] and argv[0] in HELP:
        sys.stdout.write(USAGE)
        sys.exit(0)
    if not argv or argv[0] not in COMMANDS:
        sys.stderr.write(USAGE)
        sys.exit(2)
    name, rest = argv[0], argv[1:]
    if name in NO_OPTIONS:
        if any(a in HELP for a in rest):
            sys.stdout.write(f'usage: python -m spectral_lejepa {name} {" | ".join(NO_OPTIONS[name])}\n\n'
                             + (importlib.import_module(COMMANDS[name]).__doc__ or '').strip() + '\n')
            sys.exit(0)
        if any(a not in NO_OPTIONS[name] for a in rest):
            sys.stderr.write(f'spectral_lejepa {name}: unexpected arguments {rest}\n')
            sys.exit(2)
    sys.argv = [f'spectral_lejepa {name}', *rest]   # modules parse sys.argv themselves
    importlib.import_module(COMMANDS[name]).main()


if __name__ == '__main__':
    main()
