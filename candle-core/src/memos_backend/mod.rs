//! Implementation of Backend traits for Metal
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::cpu_backend::CpuDevice;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
use std::any::type_name;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Debug, Clone)]
pub struct MemOSStorage {
    /// The actual buffer containing the data.
    pub buffer: CpuStorage,
    /// The dtype is kept since buffers are untyped.
    pub dtype: DType,
    pub device: MemOSDevice,
}

#[derive(Copy, Clone, Debug)]
pub struct MemOSDevice;

impl BackendDevice for MemOSDevice {
    type Storage = MemOSStorage;

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cpu
    }

    fn same_device(&self, _other: &Self) -> bool {
        true
    }

    fn zeros_impl(&self, shape: &crate::Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(CpuDevice.zeros_impl(shape, dtype)?.into())
    }

    unsafe fn alloc_uninit(&self, shape: &crate::Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(CpuDevice.alloc_uninit(shape, dtype)?.into())
    }

    fn storage_from_slice<T: crate::WithDType>(&self, slice: &[T]) -> Result<Self::Storage> {
        Ok(CpuDevice.storage_from_slice(slice)?.into())
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        Ok(CpuDevice.storage_from_cpu_storage(storage)?.into())
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        Ok(CpuDevice.storage_from_cpu_storage_owned(storage)?.into())
    }

    fn rand_uniform(
        &self,
        shape: &crate::Shape,
        dtype: DType,
        low: f64,
        high: f64,
    ) -> Result<Self::Storage> {
        Ok(CpuDevice.rand_uniform(shape, dtype, low, high)?.into())
    }

    fn rand_normal(
        &self,
        shape: &crate::Shape,
        dtype: DType,
        mean: f64,
        std_dev: f64,
    ) -> Result<Self::Storage> {
        Ok(CpuDevice.rand_normal(shape, dtype, mean, std_dev)?.into())
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        CpuDevice.set_seed(seed)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

impl From<CpuStorage> for MemOSStorage {
    fn from(buffer: CpuStorage) -> Self {
        Self {
            dtype: buffer.dtype(),
            buffer,
            device: MemOSDevice,
        }
    }
}

impl BackendStorage for MemOSStorage {
    type Device = MemOSDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?.into())),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?.into())),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?.into())),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?.into())),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?.into())),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?.into())),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?.into())),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().affine(layout, mul, add)?))
    }

    fn powf(&self, layout: &Layout, pow: f64) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().powf(layout, pow)?))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().elu(layout, alpha)?))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().reduce_op(op, layout, sum_dims)?))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().cmp(op, &*rhs.buffer(), lhs_l, rhs_l)?))
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        self.buffer_mut().const_set(s, l)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let buffer = self.buffer().to_dtype(layout, dtype)?;
        Ok(Self::from(buffer))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().unary_impl::<B>(layout)?))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        Ok(self.with_buffer(
            self.buffer()
                .binary_impl::<B>(&*rhs.buffer(), lhs_l, rhs_l)?,
        ))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().where_cond(
            layout,
            &*t.buffer(),
            t_l,
            &*f.buffer(),
            f_l,
        )?))
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        Ok(self.with_buffer(
            self.buffer()
                .conv1d(layout, &*kernel.buffer(), kernel_l, params)?,
        ))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        k: &Self,
        k_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().conv_transpose1d(
            layout,
            &*k.buffer(),
            k_layout,
            params,
        )?))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        Ok(self.with_buffer(
            self.buffer()
                .conv2d(layout, &*kernel.buffer(), kernel_l, params)?,
        ))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().conv_transpose2d(
            l,
            &*kernel.buffer(),
            kernel_l,
            params,
        )?))
    }

    fn avg_pool2d(
        &self,
        inp_l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        Ok(self.with_buffer(
            self.buffer()
                .avg_pool2d(inp_l, (w_k, h_k), (w_stride, h_stride))?,
        ))
    }

    fn max_pool2d(
        &self,
        inp_l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        Ok(self.with_buffer(
            self.buffer()
                .max_pool2d(inp_l, (w_k, h_k), (w_stride, h_stride))?,
        ))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().upsample_nearest1d(l, sz)?))
    }

    fn upsample_nearest2d(&self, inp_l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().upsample_nearest2d(inp_l, out_w, out_h)?))
    }

    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().gather(src_l, &*ids.buffer(), ids_l, dim)?))
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.buffer_mut()
            .scatter_set(l, &*ids.buffer(), ids_l, &*src.buffer(), src_l, dim)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.buffer_mut()
            .scatter_add_set(l, &*ids.buffer(), ids_l, &*src.buffer(), src_l, dim)
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        tracing::info!("ISM: {:p} {:p}", ids.buffer(), self.buffer());
        Ok(self.with_buffer(
            self.buffer()
                .index_select(&*ids.buffer(), src_l, ids_l, dim)?,
        ))
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        Ok(self.with_buffer(self.buffer().index_add(
            l,
            &ids.buffer(),
            ids_l,
            &src.buffer(),
            src_l,
            dim,
        )?))
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        Ok(self.with_buffer(
            self.buffer()
                .matmul(&*rhs.buffer(), (b, m, n, k), lhs_l, rhs_l)?,
        ))
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.buffer()
            .copy2d(&mut *dst.buffer_mut(), d1, d2, src_s, dst_s, src_o, dst_o)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        self.buffer()
            .copy_strided_src(&mut dst.buffer_mut(), dst_offset, src_l)
    }
}

impl MemOSStorage {
    pub fn with_buffer(&self, buffer: CpuStorage) -> Self {
        Self::new(buffer, self.device().clone(), self.dtype)
    }

    pub fn new(buffer: CpuStorage, device: MemOSDevice, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            dtype,
        }
    }

    pub fn buffer_mut(&mut self) -> &mut CpuStorage {
        &mut self.buffer
    }

    pub fn buffer(&self) -> &CpuStorage {
        &self.buffer
    }

    pub(crate) fn to_cpu<T: Clone + WithDType>(&self) -> Result<Vec<T>> {
        Ok(self.buffer().as_slice::<T>()?.to_vec())
    }

    pub fn move_to_memos(&self, ctx: &mut MemOSBuilder) -> Result<Self> {
        Ok(Self {
            buffer: self.buffer.move_to_memos(ctx)?,
            dtype: self.dtype.clone(),
            device: MemOSDevice,
        })
    }
}

pub struct MemOSBuilder {
    pub imp: Box<dyn MemOSImp>,
}

pub trait MemOSImp {
    fn alloc(&self, layout: std::alloc::Layout) -> (u128, u64, *mut u8);
    fn write_hdr(&self, base_off: u64, id: u128);
    fn get_base(&self) -> *const u8;
}

impl MemOSBuilder {
    pub fn new(imp: Box<dyn MemOSImp>) -> Self {
        Self { imp }
    }

    pub fn alloc<T>(&self, data: T) -> GPtr<T> {
        let (id, off, ptr) = self.imp.alloc(std::alloc::Layout::new::<T>());
        unsafe { ptr.cast::<T>().write(data) };
        GPtr::new(id, off)
    }

    pub fn alloc_slice<T>(&self, data: &[T]) -> GPtr<T> {
        let (id, off, ptr) = self
            .imp
            .alloc(std::alloc::Layout::array::<T>(data.len()).unwrap());
        let slice = unsafe { core::slice::from_raw_parts_mut(ptr, data.len() * size_of::<T>()) };
        let data = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), slice.len()) };
        tracing::info!(
            "alloc slice: {} {:p} {:p} {:p} {}",
            data.len(),
            slice,
            data,
            ptr,
            type_name::<T>()
        );
        slice.copy_from_slice(data);
        GPtr::new(id, off)
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct GPtr<T> {
    pub id: u128,
    pub off: u64,
    _pd: PhantomData<T>,
}

impl<T> GPtr<T> {
    pub fn cast<U>(self) -> GPtr<U> {
        GPtr {
            id: self.id,
            off: self.off,
            _pd: PhantomData,
        }
    }
}

impl<T> Copy for GPtr<T> {}
impl<T> Clone for GPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub trait Resolver {
    fn resolve_and_forget(&self, id: u128, off: u64) -> *const u8;
}

static RES: OnceLock<Box<dyn Resolver + Send + Sync + 'static>> = OnceLock::new();
static MAGIC: OnceLock<u64> = OnceLock::new();

pub(crate) fn get_resolver() -> &'static Box<dyn Resolver + Send + Sync + 'static> {
    RES.get().unwrap()
}

pub fn set_resolver(r: Box<dyn Resolver + Send + Sync + 'static>) {
    RES.set(r).map_err(|_| ()).unwrap();
    MAGIC.set(rand::random()).unwrap();
}

pub(crate) fn get_magic() -> u64 {
    *MAGIC.get().unwrap()
}

impl<T> GPtr<T> {
    pub fn new(id: u128, off: u64) -> Self {
        Self {
            id,
            off,
            _pd: PhantomData,
        }
    }

    pub fn resolve(&self) -> *const T {
        let res = RES.get().unwrap();
        let ptr = res.resolve_and_forget(self.id, self.off);
        ptr.cast()
    }
}
