from utils import save_dataset_to_hdf5, parse_args
from circles import (
    make_complex_dataset, 
    make_simple_dataset,
    generate_dataset_multiprocess,
)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    # ---- Generate Complex Dataset ----
    images, labels = generate_dataset_multiprocess(
        make_complex_dataset,
        total_samples=args.total_samples,
        n_processes=args.n_processes,
        img_size=args.img_size,
        min_n_circles=args.min_n_circles,
        max_n_circles=args.max_n_circles,
        on_balanced=args.on_balanced,
        r0=args.r0,
        min_r=args.min_r,
        max_r=args.max_r,
        box_lower_bound=args.box_lower_bound,
        box_higher_bound=args.box_higher_bound,
        max_gray=args.max_gray,
        new_center_tries=args.new_center_tries
    )
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    output_file = save_dataset_to_hdf5(
        images=images,
        labels=labels,
        output_dir=args.output_path,
        min_n_circles=args.min_n_circles,
        max_n_circles=args.max_n_circles,
        img_size=args.img_size
    )
    
    # ---- Generate Simple Dataset ----
    images, labels = generate_dataset_multiprocess(
        make_simple_dataset,
        total_samples=args.total_samples,
        n_processes=args.n_processes,
        img_size=args.img_size,
        min_n_circles=args.min_n_circles,
        max_n_circles=args.max_n_circles,
        box_lower_bound=args.box_lower_bound,
        box_higher_bound=args.box_higher_bound,
    )
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    output_file = save_dataset_to_hdf5(
        prefix="simple_images",
        images=images,
        labels=labels,
        output_dir=args.output_path,
        min_n_circles=args.min_n_circles,
        max_n_circles=args.max_n_circles,
        img_size=args.img_size
    )
