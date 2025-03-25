import collections
import os
import time
from time import sleep
import numpy as np

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime import Batch, SamplerV2

from qiskit_addon_cutting import (
    cut_wires, expand_observables,
    generate_cutting_experiments,
    partition_problem
)
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts
)

# Import Dynamical Decoupling
from src.DD.dynamical_decoupling import DynamicalDecoupling
from qiskit.circuit.library import XGate, YGate

from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

def create_dd_pass_managers(durations):
    """
    Create different Dynamical Decoupling pass managers
    
    Args:
        durations (InstructionDurations): Instruction durations for the backend
    
    Returns:
        list: List of PassManagers with different DD sequences
    """
    # Define various DD sequences
    dd_sequences = [
        # Simple X-X sequence (Hahn echo)
        [XGate(), XGate()],
        
        # Y-Y sequence
        [YGate(), YGate()],
        
        # Alternating X-Y sequence
        [XGate(), YGate()],
        
        # Uhrig dynamical decoupling (UDD) sequence
        # We'll use X gates for this example
        [XGate()] * 8  # 8-pulse sequence
    ]
    
    # Spacing types
    spacing_types = [
        None,  # balanced spacing
        # Custom spacing can be added here
    ]
    
    pass_managers = []
    
    for seq in dd_sequences:
        for spacing in spacing_types:
            # Create PassManager with DD
            pm = PassManager()
            dd_pass = DynamicalDecoupling(
                durations, 
                dd_sequence=seq, 
                spacing=spacing,
                name=f"DD_{len(seq)}_pulse_{'balanced' if spacing is None else 'custom'}"
            )
            pm.append(dd_pass)
            pass_managers.append(pm)
    
    return pass_managers

def pre_processing(logger, scale=1, qps=2, num_samples=10):    
    base_qubits = 7    
    circuit = EfficientSU2(base_qubits * scale, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)    
    observable = SparsePauliOp([i * scale for i in ["ZIIIIII", "IIIIIZI", "IIIIIIZ"]])

    # Specify settings for the cut-finding optimizer
    optimization_settings = OptimizationParameters(seed=111)

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=qps)

    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    logger.info(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.\n'
        f'Lowest cost solution found: {metadata["minimum_reached"]}.'
    )
    for cut in metadata["cuts"]:
        logger.info(f"{cut[0]} at circuit instruction index {cut[1]}")

    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    logger.info(
        f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
    )

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    
    total_subexperiments = sum(len(expts) for expts in subexperiments.values())
    logger.info(f"Total subexperiments to run on backend: {total_subexperiments}")
        
    return subexperiments, coefficients, subobservables, observable, circuit

def execute_sampler_with_dd(sampler, label, subsystem_subexpts, shots, dd_pass_managers, pass_manager):
    """
    Execute sampler with different dynamical decoupling sequences
    
    Args:
        sampler: Qiskit sampler
        label: Experiment label
        subsystem_subexpts: Subcircuits
        shots: Number of shots
        dd_pass_managers: List of PassManagers with DD sequences
        pass_manager: Original pass manager for transpilation
    
    Returns:
        List of results with different DD sequences
    """
    results = []
    
    for dd_pm in dd_pass_managers:
        # Combine original transpilation with DD
        combined_pm = PassManager()
        combined_pm.append(pass_manager)
        combined_pm.append(dd_pm)
        
        # Apply transpilation and DD to subcircuits
        dd_subcircuits = [combined_pm.run(circ) for circ in subsystem_subexpts]
        
        # Run the modified circuits
        job = sampler.run(dd_subcircuits, shots=shots)
        result = job.result()
        
        results.append((label, result))
    
    return results

def main():
    # Similar setup as in the original script
    pilot_compute_description_ray = {
        "resource": "slurm://localhost",
        "working_directory": os.path.join(os.environ["HOME"], "work"),
        "number_of_nodes": 1,
        "cores_per_node": 8,
        "gpus_per_node": 2,
        "queue": "debug",
        "walltime": 30,
        "type": "ray",
        "scheduler_script_commands": ["#SBATCH --partition=gpua16","#SBATCH --gres=gpu:2"]
    }

    pcs = None
    try:
        # Start Pilot Compute Service
        pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY)
        pcs.create_pilot(pilot_compute_description_ray).wait()
        logger = pcs.get_logger()

        # Preprocessing
        subexperiments, coefficients, subobservables, observable, circuit = pre_processing(logger)

        # Backend setup
        backend_options = {
            "backend_options": {
                "shots": 4096, 
                "device": "GPU", 
                "method": "statevector", 
                "blocking_enable": True, 
                "batched_shots_gpu": True, 
                "blocking_qubits": 25
            }
        }
        backend = AerSimulator(**backend_options["backend_options"])

        # Create pass managers
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        
        # Create instruction durations (you might want to customize this)
        durations = InstructionDurations([
            ('x', None, 50),  # default X gate duration
            ('y', None, 50),  # default Y gate duration
        ])
        
        # Create DD pass managers
        dd_pass_managers = create_dd_pass_managers(durations)

        # Transpile subcircuits
        isa_subexperiments = {
            label: pass_manager.run(partition_subexpts)
            for label, partition_subexpts in subexperiments.items()
        }

        # Execute with different DD sequences
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            all_results = []
            
            for label, subsystem_subexpts in isa_subexperiments.items():
                for ss in subsystem_subexpts:
                    results = pcs.submit_task(
                        execute_sampler_with_dd, 
                        sampler, 
                        label, 
                        [ss], 
                        shots=2**12, 
                        dd_pass_managers=dd_pass_managers,
                        pass_manager=pass_manager,
                        resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                    )
                    all_results.append(results)

            # Collect and process results
            results_tuple = pcs.get_results(all_results)

# Process results from different DD sequences
        samplePubResults = collections.defaultdict(list)
        dd_results = {}  # Store results for each DD sequence

        # Organize results by DD sequence and label
        for result_tuple in results_tuple:
            label, result = result_tuple
            dd_sequence_index = results_tuple.index(result_tuple) // len(isa_subexperiments)
            
            # Collect results
            samplePubResults[label].extend(result._pub_results)
            
            # Store results for each DD sequence
            if label not in dd_results:
                dd_results[label] = []
            dd_results[label].append(result)

        # Process results for each DD sequence
        dd_reconstructed_expvals = {}
        
        for dd_seq_name, samples in samplePubResults.items():
            results = {dd_seq_name: PrimitiveResult(samples)}
            
            # Reconstruct expectation values
            reconstructed_expvals = reconstruct_expectation_values(
                results,
                coefficients,
                subobservables,
            )
            
            # Calculate final expectation value
            final_expval = np.dot(reconstructed_expvals, observable.coeffs)
            dd_reconstructed_expvals[dd_seq_name] = final_expval

        # Transpile full circuit
        transpile_full_circuit_time = time.time()
        full_circuit_transpilation = pass_manager.run(circuit)
        transpile_full_circuit_end_time = time.time()
        logger.info(f"Execution time for full Circuit transpilation: {transpile_full_circuit_end_time-transpile_full_circuit_time}")

        # Run full circuit to get exact expectation value
        exact_expval = 0
        for i in range(3):
            full_circuit_estimator_time = time.time()                           
            full_circuit_task = pcs.submit_task(
                run_full_circuit, 
                observable, 
                backend_options, 
                full_circuit_transpilation, 
                resources={'num_cpus': 1, 'num_gpus': 2, 'memory': None}
            )
            exact_expval = pcs.get_results([full_circuit_task])[0]
            full_circuit_estimator_end_time = time.time()
            
            logger.info(f"Execution time for full Circuit: {full_circuit_estimator_end_time-full_circuit_estimator_time}")

        # Compare results for each DD sequence
        logger.info("Dynamical Decoupling Results Comparison:")
        logger.info(f"Exact expectation value: {np.round(exact_expval, 8)}")
        
        for dd_seq_name, reconstructed_expval in dd_reconstructed_expvals.items():
            logger.info(f"\nDD Sequence: {dd_seq_name}")
            logger.info(f"Reconstructed expectation value: {np.real(np.round(reconstructed_expval, 8))}")
            logger.info(f"Error in estimation: {np.real(np.round(reconstructed_expval - exact_expval, 8))}")
            logger.info(
                f"Relative error in estimation: {np.real(np.round((reconstructed_expval - exact_expval) / exact_expval, 8))}"
            )

    except Exception as e:
        logger.error(f"Error in execution: {e}")
    finally:
        if pcs:
            pcs.cancel()

if __name__ == "__main__":
    main()
