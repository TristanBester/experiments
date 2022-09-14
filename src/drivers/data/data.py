from torch.utils.data import DataLoader

from ...datasets import (
    BeetleFlyDataset,
    BirdChickenDataset,
    ComputersDataset,
    EarthquakesDataset,
    ItalyPowerDemandDataset,
    MoteStrainDataset,
    PhalangesOutlinesCorrectDataset,
    ProximalPhalanxOutlineCorrectDataset,
    ShapeletSimDataset,
    SonyAIBORobotSurfaceDataset,
    SonyAIBORobotSurfaceIIDataset,
    WormsTwoClassDataset,
)


def init_datasets(base_path, device):
    datasets = []

    datasets.append(
        BeetleFlyDataset(
            train_path=f"{base_path}/BeetleFly/BeetleFly_TRAIN",
            test_path=f"{base_path}/BeetleFly/BeetleFly_TEST",
            device=device,
        )
    )
    datasets.append(
        BirdChickenDataset(
            train_path=f"{base_path}/BirdChicken/BirdChicken_TRAIN",
            test_path=f"{base_path}/BirdChicken/BirdChicken_TEST",
            device=device,
        )
    )
    datasets.append(
        ComputersDataset(
            train_path=f"{base_path}/Computers/Computers_TRAIN",
            test_path=f"{base_path}/Computers/Computers_TEST",
            device=device,
        )
    )
    datasets.append(
        EarthquakesDataset(
            train_path=f"{base_path}/Earthquakes/Earthquakes_TRAIN",
            test_path=f"{base_path}/Earthquakes/Earthquakes_TEST",
            device=device,
        )
    )
    datasets.append(
        ItalyPowerDemandDataset(
            train_path=f"{base_path}/ItalyPowerDemand/ItalyPowerDemand_TRAIN",
            test_path=f"{base_path}/ItalyPowerDemand/ItalyPowerDemand_TEST",
            device=device,
        )
    )
    datasets.append(
        MoteStrainDataset(
            train_path=f"{base_path}/MoteStrain/MoteStrain_TRAIN",
            test_path=f"{base_path}/MoteStrain/MoteStrain_TEST",
            device=device,
        )
    )
    datasets.append(
        PhalangesOutlinesCorrectDataset(
            train_path=f"{base_path}/PhalangesOutlinesCorrect/PhalangesOutlinesCorrect_TRAIN",
            test_path=f"{base_path}/PhalangesOutlinesCorrect/PhalangesOutlinesCorrect_TEST",
            device=device,
        )
    )
    datasets.append(
        ProximalPhalanxOutlineCorrectDataset(
            train_path=f"{base_path}/ProximalPhalanxOutlineCorrect/ProximalPhalanxOutlineCorrect_TRAIN",
            test_path=f"{base_path}/ProximalPhalanxOutlineCorrect/ProximalPhalanxOutlineCorrect_TEST",
            device=device,
        )
    )
    datasets.append(
        ShapeletSimDataset(
            train_path=f"{base_path}/ShapeletSim/ShapeletSim_TRAIN",
            test_path=f"{base_path}/ShapeletSim/ShapeletSim_TEST",
            device=device,
        )
    )
    datasets.append(
        SonyAIBORobotSurfaceDataset(
            train_path=f"{base_path}/SonyAIBORobotSurface/SonyAIBORobotSurface_TRAIN",
            test_path=f"{base_path}/SonyAIBORobotSurface/SonyAIBORobotSurface_TEST",
            device=device,
        )
    )
    datasets.append(
        SonyAIBORobotSurfaceIIDataset(
            train_path=f"{base_path}/SonyAIBORobotSurfaceII/SonyAIBORobotSurfaceII_TRAIN",
            test_path=f"{base_path}/SonyAIBORobotSurfaceII/SonyAIBORobotSurfaceII_TEST",
            device=device,
        )
    )
    datasets.append(
        WormsTwoClassDataset(
            train_path=f"{base_path}/WormsTwoClass/WormsTwoClass_TRAIN",
            test_path=f"{base_path}/WormsTwoClass/WormsTwoClass_TEST",
            device=device,
        )
    )
    return datasets


def init_data(base_path, batch_size, device):
    datasets = init_datasets(base_path, device=device)
    loaders = [DataLoader(dataset=d, batch_size=batch_size) for d in datasets]
    return list(zip(datasets, loaders))

