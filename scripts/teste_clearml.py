#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
from clearml import Task, Dataset


def get_task_if_remote():
    task_id = os.environ.get("CLEARML_TASK_ID")
    if task_id is not None:
        return Task.get_task(task_id=task_id)
    return None


def main():
    task_evaluate = get_task_if_remote()
    if task_evaluate is None:
        task_evaluate = Task.init(
            project_name="teste_guto",
            task_name=f"3dgs_0407_pro",
            task_type=Task.TaskTypes.training,
        )
        task_evaluate.set_base_docker(
            docker_image="10.167.1.54:81/nerfs/gaussian-splatting-ngp:latest",
            docker_arguments="--shm-size=128000mb  --memory=64g ",
        )
        task_evaluate.execute_remotely(queue_name="default")

    # Acessando o dataset usando o ClearML
    dataset = Dataset.get(dataset_id="ce9bb6dd898a45b2830b0a6a4ed031ac")

    # Baixando a cópia local do dataset
    dataset_path = dataset.get_local_copy()
    dataset_path = Path(dataset_path)
    print(f"\nDataset path: {dataset_path}")
    specific_folder = dataset_path / "04_07/pro/Gsplat"
    specific_folder = Path(specific_folder)

    print(f"\nCaminho final: {specific_folder}")
    output_dir = "/app/output"

    # Encontrando o arquivo train.py
    # os.system("find / -name train.py")
    # os.system("ls -l /app")

    # # Executando o script de treinamento com o caminho do dataset usando subprocess
    # command = ["python3", "/app/train.py", "-s", str(specific_folder)]
    # print(f"Running command: {' '.join(command)}")

    # # Executa o comando e redireciona a saída em tempo real
    # process = subprocess.Popen(
    #     command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    # )

    # # Log da saída em tempo real
    # for line in process.stdout:
    #     print(line, end="")  # Loga cada linha de stdout

    # for line in process.stderr:
    #     print(line, end="")  # Loga cada linha de stderr

    # process.wait()  # Espera o processo terminar

    # if task_evaluate:
    #     task_evaluate.close()

    # Executando o script de treinamento com o caminho do dataset
    command = f"python3 /app/train.py -s {specific_folder} --iterations 90000 --save_iterations 90000 -r 800 -m {output_dir}/out_1"
    print(f"Running command: {command}")
    os.system(command)

    # Upload do diretório de saída como um artefato no ClearML
    if task_evaluate:
        task_evaluate.upload_artifact("output_directory", str(output_dir))
        print(f"Training completed successfully. Outputs saved in {output_dir}")
        task_evaluate.close()


if __name__ == "__main__":
    main()
