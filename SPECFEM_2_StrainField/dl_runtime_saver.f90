!=====================================================================
! Write out the strain field.
! Added by Liang Ding 
!=====================================================================

  subroutine dl_runtime_save_strain(idx_step)
  ! save the strain field during running
   
    use specfem_par
    use specfem_par_elastic
    use specfem_par_acoustic
    use specfem_par_poroelastic
   
    implicit none
    integer :: idx_step
    integer :: ier 
    character(len=128) :: str_file_name
    character(len=20) :: str_idx_step = '1'

    ! create file name
    Write(str_idx_step, '(I10)') idx_step
    str_file_name = prname(1:len_trim(prname))//trim('strain_field_Step_')//trim(adjustl(str_idx_step))//trim('.bin')

    ! save strain field.
    if (SIMULATION_TYPE == 1 .and. SAVE_FORWARD) then
        open(unit=IOUT,file=trim(str_file_name), &
              status='unknown',form='unformatted',iostat=ier)
                 
        if (ier /= 0) then
          print *,'error: opening save_forward_arrays.bin'
          print *,'path: ',trim(str_file_name)
          call exit_mpi(myrank,'error opening file save_forward_arrays.bin')
        endif
         
        write(IOUT) epsilondev_trace
        write(IOUT) epsilondev_xx
        write(IOUT) epsilondev_yy
        write(IOUT) epsilondev_xy
        write(IOUT) epsilondev_xz
        write(IOUT) epsilondev_yz
        FLUSH(IOUT)
        close(IOUT)
    endif

  end subroutine dl_runtime_save_strain


