use super::GgmlDType;
use crate::{
    backend::{BackendDevice, BackendStorage},
    quantized::QStorage,
    wgpu_backend::{
        wgpu_functions::{
            self,
            matmul::{
                sgemm::{
                    GenericDynamicMatmulShaderSettings, GenericMatmulSettings, StrideOptimization,
                },
                SGEMMParams,
            },
            QueueLayouts, WgpuTensor,
        },
        QuantizedMatmulAlgorithm,
    },
    DType, Result, Shape, WgpuDevice, WgpuStorage,
};
use wgpu_compute_layer::cache::BufferReferenceId;

pub struct QWgpuStorage {
    dtype: GgmlDType,
    storage: WgpuStorage,
}

impl QWgpuStorage {
    pub fn new(dtype: GgmlDType, storage: WgpuStorage) -> Self {
        Self { dtype, storage }
    }
    pub fn buffer(&self) -> BufferReferenceId {
        self.storage.buffer()
    }
    pub fn zeros(device: &WgpuDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        Ok(QWgpuStorage::new(
            dtype,
            device.zeros_impl(&(size / 4,).into(), DType::U32)?,
        ))
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &WgpuDevice {
        self.storage.device()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<WgpuStorage> {
        let dev = self.device();
        let dst = dev.alloc_uninit_size(DType::F32, elem_count);

        if self.dtype == GgmlDType::F32 {
            //no need to dequantize
            wgpu_functions::queue_copy(
                dev,
                dst.buffer(),
                self.storage.buffer(),
                0,
                0,
                self.storage.size_in_bytes() / 4,
                DType::U32,
            )?;
            return Ok(dst);
        }

        let mut queue = dev.get_queue();
        queue.add(elem_count);
        let pipeline = match self.dtype() {
            GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q4_0::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q4_1::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q5_0::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q5_1::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q8_0::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q8_1::Functions::DequantizeBlockToF32,
            ),

            GgmlDType::Q2K => candle_wgpu_kernels::Pipelines::Q2K(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q2_k::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q3K => candle_wgpu_kernels::Pipelines::Q3K(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q3_k::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q4K => candle_wgpu_kernels::Pipelines::Q4K(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q4_k::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q5K => candle_wgpu_kernels::Pipelines::Q5K(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q5_k::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q6K => candle_wgpu_kernels::Pipelines::Q6K(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q6_k::Functions::DequantizeBlockToF32,
            ),
            GgmlDType::Q8K => candle_wgpu_kernels::Pipelines::Q8K(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q8_k::Functions::DequantizeBlockToF32,
            ),
            _ => {
                crate::bail!("Dequantize not implemented for {:?}", self.dtype());
            }
        };
        let pipeline = queue.get_pipeline(pipeline);
        let bind_group =
            dev.create_bind_group_input1(dst.buffer(), self.buffer(), DType::F32.into());
        queue.enqueue_64(
            pipeline,
            bind_group,
            (elem_count / self.dtype().block_size()) as u32,
            elem_count,
        );

        Ok(dst)
    }

    pub fn quantize(&mut self, src: &WgpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu_storage()?;
        let elem_count = src.as_slice::<f32>()?.len();
        let src = crate::Storage::Cpu(src);
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self
            .device()
            .alloc_from_slice(DType::U32, &qcpu_storage.data()?)?;
        self.storage = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self
            .device()
            .alloc_from_slice(DType::U32, &qcpu_storage.data()?)?;
        self.storage = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &WgpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu_storage()?;
        let elem_count = src.as_slice::<f32>()?.len();
        let src = crate::Storage::Cpu(src);
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self
            .device()
            .alloc_from_slice(DType::U32, &qcpu_storage.data()?)?;
        self.storage = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self
            .device()
            .alloc_from_slice(DType::U32, &qcpu_storage.data()?)?;
        self.storage = buffer;
        Ok(())
    }


    fn get_best_algorithm(
        &self,
        dtype: GgmlDType,
        (_, m, n, k): (usize, usize, usize, usize),
        input1_stride_k: usize,
    ) -> QuantizedMatmulAlgorithm {
        match dtype {
            GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1 => {
                if k % 32 == 0 && m % 32 == 0 && n % 32 == 0 {
                    //the fastes configuration seen in benchmarks on q8_0:
                    if m % 128 == 0 && n % 64 == 0 {
                        QuantizedMatmulAlgorithm::Some(GenericDynamicMatmulShaderSettings::new(
                            GenericMatmulSettings::new(
                                128,
                                64,
                                32,
                                StrideOptimization::None,
                                StrideOptimization::StrideK(true),
                            ),
                            16,
                            4,
                            false,
                        ))
                    } else if m % 64 == 0 && n % 128 == 0 {
                        QuantizedMatmulAlgorithm::Some(GenericDynamicMatmulShaderSettings::new(
                            GenericMatmulSettings::new(
                                64,
                                128,
                                32,
                                StrideOptimization::None,
                                StrideOptimization::StrideK(true),
                            ),
                            16,
                            2,
                            false,
                        ))
                    } else if m % 64 == 0 && n % 64 == 0 {
                        QuantizedMatmulAlgorithm::Some(GenericDynamicMatmulShaderSettings::new(
                            GenericMatmulSettings::new(
                                64,
                                64,
                                32,
                                StrideOptimization::None,
                                StrideOptimization::StrideK(true),
                            ),
                            8,
                            4,
                            false,
                        ))
                    } else if m % 32 == 0 && n % 64 == 0 {
                        QuantizedMatmulAlgorithm::Some(GenericDynamicMatmulShaderSettings::new(
                            GenericMatmulSettings::new(
                                32,
                                64,
                                32,
                                StrideOptimization::None,
                                StrideOptimization::StrideK(true),
                            ),
                            4,
                            4,
                            false,
                        ))
                    } else if m % 64 == 0 && n % 32 == 0 {
                        QuantizedMatmulAlgorithm::Some(GenericDynamicMatmulShaderSettings::new(
                            GenericMatmulSettings::new(
                                64,
                                32,
                                32,
                                StrideOptimization::None,
                                StrideOptimization::StrideK(true),
                            ),
                            8,
                            4,
                            false,
                        ))
                    } else {
                        QuantizedMatmulAlgorithm::Some(GenericDynamicMatmulShaderSettings::new(
                            GenericMatmulSettings::new(
                                32,
                                32,
                                32,
                                StrideOptimization::None,
                                StrideOptimization::StrideK(true),
                            ),
                            8,
                            2,
                            false,
                        ))
                    }
                } else if m == 1 && k % 32 == 0 && input1_stride_k == 1 {
                    match dtype {
                        GgmlDType::Q8_1 => {
                            if n % 128 == 0 {
                                QuantizedMatmulAlgorithm::Some(
                                    GenericDynamicMatmulShaderSettings::new_tiled_small(
                                        GenericMatmulSettings::new(
                                            1,
                                            32,
                                            128,
                                            StrideOptimization::StrideK(true),
                                            StrideOptimization::StrideK(true),
                                        ),
                                        1,
                                        32,
                                        false,
                                    ),
                                )
                            } else {
                                QuantizedMatmulAlgorithm::Naive
                            }
                        }
                        _ => QuantizedMatmulAlgorithm::Naive,
                    }
                } else {
                    QuantizedMatmulAlgorithm::Naive
                }
            }
            _ => QuantizedMatmulAlgorithm::Naive,
        }
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(WgpuStorage, Shape)> {
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let src_shape = src_shape.dims().to_vec();

        let (b, m) = match src_shape.len() {
            3 => (src_shape[0], src_shape[1]),
            2 => (1, src_shape[0]),
            n => crate::bail!("Invalid rank {n} for quantized matmul wgpu"),
        };
        let mut dst_shape = src_shape;
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);

        let mut input1_stride = layout.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);

        let dst_shape = Shape::from(dst_shape);
        let dev = storage.device();
        let dst = dev.alloc_uninit_size(DType::F32, dst_shape.elem_count());

        let matmul_alg = dev
            .inner_device()
            .with_extension::<QuantizedMatmulAlgorithm, QuantizedMatmulAlgorithm>(|c| c.clone())
            .unwrap_or(QuantizedMatmulAlgorithm::None);
        let matmul_alg: QuantizedMatmulAlgorithm = match &matmul_alg {
            QuantizedMatmulAlgorithm::None => {
                self.get_best_algorithm(self.dtype, (b, m, n, k), input1_stride_k)
            }
            QuantizedMatmulAlgorithm::Naive => QuantizedMatmulAlgorithm::Naive,
            QuantizedMatmulAlgorithm::Some(setting) => {
                QuantizedMatmulAlgorithm::Some(setting.to_owned())
            }
        };

        // Int-domain Q4_K × Q8K matmul fast path (decode hot path).
        // Conditions: m==1, dtype=Q4K, DP4A available, contiguous f32
        // activation row, k a multiple of 256 (Q4_K block size).
        // When all hold, we run a Q8K quantize pre-pass over the
        // activation buffer and dispatch the int-domain matmul. Drift
        // drops from ~0.5% per matmul (f32-dequant kernel) to <1e-5.
        let int_q4k_eligible = matches!(matmul_alg, QuantizedMatmulAlgorithm::Naive)
            && m == 1
            && b == 1
            && self.dtype() == GgmlDType::Q4K
            && dev.inner_device().supports_dp4a()
            && k % 256 == 0
            && input1_stride_k == 1
            && layout.start_offset() == 0;

        if int_q4k_eligible {
            // Pre-pass: quantize the f32 activation row to Q8K. The
            // returned `WgpuStorage` owns the transient block buffer;
            // we only need its `BufferReferenceId` for the bind group.
            let q8k_storage =
                quantize_row_to_q8k(dev, storage.buffer(), k * m * b)?;
            let q8k_buf = q8k_storage.buffer();
            let mut queue = dev.get_queue();
            queue.add(m);
            queue.add(k);
            queue.add(n);
            queue.add(input1_stride_b);
            queue.add(layout.start_offset());
            queue.add(input1_stride_k);
            queue.add(input1_stride_m);
            let pipeline = candle_wgpu_kernels::Pipelines::Q4kInt(
                candle_wgpu_kernels::DType::F32,
                candle_wgpu_kernels::quantized::q4k_int::Functions::MatmulQ4kQ8kM1,
            );
            let pipeline = queue.get_pipeline(pipeline);
            let bind_group = dev.create_bind_group_input2(
                dst.buffer(),
                q8k_buf,
                self.buffer(),
                DType::F32.into(),
            );
            queue.enqueue_workgroups_extra(
                pipeline,
                bind_group,
                n as u32,
                1,
                1,
                k * m * n * b,
                #[cfg(feature = "wgpu_debug")]
                Some(wgpu_functions::matmul::sgemm::get_debug_string(
                    &SGEMMParams::new(b, m, k, n),
                )),
            );
            return Ok((dst, dst_shape));
        }

        match matmul_alg {
            QuantizedMatmulAlgorithm::Naive => {
                //naive matmul

                let mut queue = dev.get_queue();
                //queue.add(b);
                queue.add(m);
                queue.add(k);
                queue.add(n);

                queue.add(input1_stride_b); //input1_stride_b
                queue.add(layout.start_offset()); //input1_offset
                                                  //queue.add(0); //input2_stride_b
                                                  //queue.add(0); //input2_ofset
                queue.add(input1_stride_k);
                queue.add(input1_stride_m);

                if m == 1 {
                    let pipeline = match self.dtype() {
                        GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q4_0::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q4_1::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q5_0::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q5_1::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q8_0::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q8_1::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q2K => candle_wgpu_kernels::Pipelines::Q2K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q2_k::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q3K => candle_wgpu_kernels::Pipelines::Q3K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q3_k::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q4K => candle_wgpu_kernels::Pipelines::Q4K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q4_k::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q5K => candle_wgpu_kernels::Pipelines::Q5K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q5_k::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q6K => candle_wgpu_kernels::Pipelines::Q6K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q6_k::Functions::MatmulNaiveBlockM1,
                        ),
                        GgmlDType::Q8K => candle_wgpu_kernels::Pipelines::Q8K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q8_k::Functions::MatmulNaiveBlockM1,
                        ),
                        _ => todo!(),
                    };

                    let const_vec = vec![
                        (input1_stride_k == 1) as usize,
                        (input1_stride_m == 1) as usize,
                        (b != 1) as usize,
                    ];

                    let pipeline = queue.get_pipeline_const(pipeline, const_vec);
                    let bind_group = dev.create_bind_group_input2(
                        dst.buffer(),
                        storage.buffer(),
                        self.buffer(),
                        DType::F32.into(),
                    );

                    queue.enqueue_workgroups_extra(
                        pipeline,
                        bind_group,
                        (n as u32).div_ceil(32),
                        1,
                        b as u32,
                        k * m * n * b,
                        #[cfg(feature = "wgpu_debug")]
                        Some(wgpu_functions::matmul::sgemm::get_debug_string(
                            &SGEMMParams::new(b, m, k, n),
                        )),
                    );
                } else {
                    let pipeline = match self.dtype() {
                        GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q4_0::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q4_1::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q5_0::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q5_1::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q8_0::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q8_1::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q2K => candle_wgpu_kernels::Pipelines::Q2K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q2_k::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q3K => candle_wgpu_kernels::Pipelines::Q3K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q3_k::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q4K => candle_wgpu_kernels::Pipelines::Q4K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q4_k::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q5K => candle_wgpu_kernels::Pipelines::Q5K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q5_k::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q6K => candle_wgpu_kernels::Pipelines::Q6K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q6_k::Functions::MatmulNaiveBlock,
                        ),
                        GgmlDType::Q8K => candle_wgpu_kernels::Pipelines::Q8K(
                            candle_wgpu_kernels::DType::F32,
                            candle_wgpu_kernels::quantized::q8_k::Functions::MatmulNaiveBlock,
                        ),
                        _ => todo!(),
                    };

                    let const_vec = vec![
                        (input1_stride_k == 1) as usize,
                        (input1_stride_m == 1) as usize,
                        (b != 1) as usize,
                    ];

                    let pipeline = queue.get_pipeline_const(pipeline, const_vec);
                    let bind_group = dev.create_bind_group_input2(
                        dst.buffer(),
                        storage.buffer(),
                        self.buffer(),
                        DType::F32.into(),
                    );

                    queue.enqueue_workgroups_extra(
                        pipeline,
                        bind_group,
                        (n as u32).div_ceil(16),
                        (m as u32).div_ceil(16),
                        b as u32,
                        k * m * n * b,
                        #[cfg(feature = "wgpu_debug")]
                        Some(wgpu_functions::matmul::sgemm::get_debug_string(
                            &SGEMMParams::new(b, m, k, n),
                        )),
                    );
                }
            }
            QuantizedMatmulAlgorithm::Some(generic_dynamic_matmul_shader_settings) => {
                let pipeline = match self.dtype() {
                    GgmlDType::Q4_0 => candle_wgpu_kernels::Pipelines::Q40(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q4_0::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q4_1 => candle_wgpu_kernels::Pipelines::Q41(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q4_1::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q5_0 => candle_wgpu_kernels::Pipelines::Q50(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q5_0::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q5_1 => candle_wgpu_kernels::Pipelines::Q51(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q5_1::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q8_0 => candle_wgpu_kernels::Pipelines::Q80(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q8_0::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q8_1 => candle_wgpu_kernels::Pipelines::Q81(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q8_1::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q2K => candle_wgpu_kernels::Pipelines::Q2K(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q2_k::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q3K => candle_wgpu_kernels::Pipelines::Q3K(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q3_k::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q4K => candle_wgpu_kernels::Pipelines::Q4K(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q4_k::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q5K => candle_wgpu_kernels::Pipelines::Q5K(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q5_k::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q6K => candle_wgpu_kernels::Pipelines::Q6K(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q6_k::Functions::MatmulSgemm,
                    ),
                    GgmlDType::Q8K => candle_wgpu_kernels::Pipelines::Q8K(
                        candle_wgpu_kernels::DType::F32,
                        candle_wgpu_kernels::quantized::q8_k::Functions::MatmulSgemm,
                    ),
                    _ => todo!(),
                };

                wgpu_functions::matmul::sgemm::queue_matmul_quantized(
                    dev,
                    dst.buffer(),
                    WgpuTensor::new(layout, storage.buffer()),
                    WgpuTensor::new(
                        &crate::Layout::new(self_shape.clone(), [1, k].to_vec(), 0),
                        self.storage.buffer(),
                    ),
                    SGEMMParams::new(b, m, k, n),
                    pipeline,
                    &generic_dynamic_matmul_shader_settings,
                )?;
            }
            QuantizedMatmulAlgorithm::None => panic!(),
        }

        Ok((dst, dst_shape))
    }

    pub async fn data_async(&self) -> Result<Vec<u8>> {
        Ok(self.storage.0.read_from_buffer_reference_async().await?)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.data_async())
        }
        #[cfg(target_arch = "wasm32")]
        {
            crate::bail!("Synchronous read not supported on wasm32");
        }
    }
}

pub fn load_quantized(device: &WgpuDevice, dtype: GgmlDType, data: &[u8]) -> Result<QStorage> {
    let storage = device.alloc_from_bytes(DType::U8, data)?;
    Ok(QStorage::Wgpu(QWgpuStorage { dtype, storage }))
}

/// Quantize a contiguous f32 row to BlockQ8K format on the WGPU device.
///
/// GPU-side equivalent of CPU's `BlockQ8K::from_float(xs, ys)`. Produces
/// a byte-identical result — see `kernels/quantized/quantize_q8k.pwgsl`
/// for the shader and the numeric-correctness notes there.
///
/// `src` must point to a contiguous `[f32; total_elems]` buffer.
/// `total_elems` MUST be a multiple of `QK_K` (256); the caller is
/// responsible for any padding. The returned `WgpuStorage` is
/// `n_blocks * 73` u32 words (i.e. `n_blocks * 292` bytes) and matches
/// `BlockQ8K`'s `#[repr(C)]` layout byte-for-byte. This is the
/// activation operand for the int-domain Q4_K matmul (Step 4 of the
/// int-domain port plan).
pub fn quantize_row_to_q8k(
    device: &WgpuDevice,
    src: BufferReferenceId,
    total_elems: usize,
) -> Result<WgpuStorage> {
    if total_elems % crate::quantized::k_quants::QK_K != 0 {
        crate::bail!(
            "quantize_row_to_q8k: total_elems ({}) must be a multiple of QK_K ({})",
            total_elems,
            crate::quantized::k_quants::QK_K
        );
    }
    let n_blocks = total_elems / crate::quantized::k_quants::QK_K;
    // Each BlockQ8K is 292 bytes = 73 u32 words. Allocate the dest as
    // U32 so the binding alignment lines up with `array<u32>` in WGSL.
    let dst_words = n_blocks * 73;
    let dst = device.alloc_uninit_size(DType::U32, dst_words);

    let mut queue = device.get_queue();
    queue.add(total_elems as u32);

    let pipeline = candle_wgpu_kernels::Pipelines::QuantizeQ8k(
        candle_wgpu_kernels::DType::F32,
        candle_wgpu_kernels::quantized::quantize_q8k::Functions::QuantizeRowToQ8k,
    );
    let pipeline = queue.get_pipeline(pipeline);

    // Both bindings are 4-byte aligned (f32 input, u32 output).
    let bind_group = device.create_bind_group_input1(
        dst.buffer(),
        src,
        DType::F32.into(),
    );

    // One workgroup per Q8K block. workgroup_size in WGSL is (64,1,1).
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        n_blocks as u32,
        1,
        1,
        total_elems,
    );

    Ok(dst)
}
