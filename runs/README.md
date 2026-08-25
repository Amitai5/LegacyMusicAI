# Run storage

Training, evaluation, voice-test, and generation runs are immutable runtime artifacts and are ignored by Git. Every run receives a unique directory containing its request, state, logs, subprocess records, provenance, and outputs.

Failed runs are retained for diagnosis and resume. Reproduction creates a new child run rather than overwriting its parent.
