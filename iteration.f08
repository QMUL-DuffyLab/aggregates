program iteration
  use iso_fortran_env
  use iso_c_binding
  implicit none
  integer, parameter :: dp = c_double
  integer, parameter :: short = c_short
  integer, parameter :: ip = c_int
  real, parameter :: pi = 3.1415926535

  character(len=200) :: params_file, rates_file, neighbours_file,&
    loss_file, seed_str, cols, counts_file, prefix_long
  character(len=:), allocatable :: file_path, prefix
  logical(c_bool), dimension(:), allocatable :: is_quencher
  integer(ip) :: i, j, n_iter, n_sites, max_neighbours, rate_size,&
    n_current, seed_size, stat, col, max_count, n_quenchers
  integer(ip), dimension(:), allocatable :: n_i, n_pq, n_q, quenchers, seed
  integer(ip), dimension(:), allocatable :: neighbours_temp, n_gen,&
    n_ann, n_po_d, n_pq_d, n_q_d, ann_bin, pool_bin, pq_bin, q_bin
  integer(ip), dimension(:, :), allocatable :: neighbours
  real(dp) :: mu, fluence, t, dt, t_pulse, rho_q,&
    sigma, fwhm, start_time, end_time, binwidth, max_time
  real(dp), dimension(:), allocatable :: rates, base_rates, pulse,&
    bins

  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  call get_command_argument(1, params_file)
  open(file=params_file, unit=20)
  read(20, *) n_sites
  read(20, *) max_neighbours
  read(20, *) rho_q
  read(20, *) fluence
  read(20, *) mu
  read(20, *) fwhm
  read(20, *) binwidth
  read(20, *) max_count
  read(20, '(a)') rates_file
  read(20, '(a)') neighbours_file
  close(20)
  rate_size = max_neighbours + 5
  write(*, '(a, I4)')     "n_sites    = ", n_sites
  write(*, '(a, I1)')     "max neigh  = ", max_neighbours
  write(*, '(a, F8.3)')   "rho_q      = ", rho_q
  write(*, '(a, ES10.3)') "fluence    = ", fluence
  write(*, '(a, F8.3)')   "mu         = ", mu
  write(*, '(a, F8.3)')   "fwhm       = ", fwhm
  write(*, '(a, F8.3)')   "binwidth   = ", binwidth
  write(*, '(a, I8)')     "max count  = ", max_count
  write(*, '(a, a)')      "rate file  = ", rates_file
  write(*, '(a, a)')      "neigh file = ", neighbours_file

  i = scan(rates_file, "/\", .true.)
  file_path = rates_file(:i)
  write(prefix_long, '(F4.2, a, ES8.2, a)') rho_q, "_", fluence, "_"
  prefix = trim(adjustl(prefix_long))
  write(*, *) "File path = ", file_path
  write(*, *) "Prefix = ", prefix
  write(loss_file, '(a, a, a)') file_path,&
    prefix, "decays.dat"
  write(counts_file, '(a, a, a)') file_path,&
    prefix, "counts.dat"

  ! fortran's fine with 0-sized arrays so this is okay
  allocate(neighbours_temp(n_sites * max_neighbours))
  allocate(neighbours(n_sites, max_neighbours))
  if (max_neighbours.ne.0) then
    open(file=neighbours_file, unit=20)
    read(20, *) neighbours_temp
    close(20)
  end if
  neighbours = reshape(neighbours_temp, (/n_sites, max_neighbours/))
  allocate(base_rates((n_sites + 2) * rate_size))
  allocate(rates(rate_size))
  allocate(n_i(n_sites))
  allocate(n_pq(n_sites))
  allocate(n_q(n_sites))

  dt = 1.0_dp
  t_pulse = 2.0_dp * mu
  allocate(pulse(int(t_pulse / dt)))
  sigma = fwhm / (2.0_dp * (sqrt(2.0_dp * log(2.0_dp))))
  do i = 1, int(t_pulse / dt)
    pulse(i) = 1.0_dp / (sigma * sqrt(2.0_dp * pi)) * &
      exp(-1.0_dp * ((i * dt) - mu)**2 / (sqrt(2.0_dp) * sigma)**2)
  end do

  ! one set of bins for each decay type; the bin array index will be
  ! something like floor(t / bin_size) + 1
  max_time = 10000.0_dp
  allocate(bins(int(max_time/binwidth)))
  allocate(ann_bin(int(max_time/binwidth)))
  allocate(pool_bin(int(max_time/binwidth)))
  allocate(pq_bin(int(max_time/binwidth)))
  allocate(q_bin(int(max_time/binwidth)))
  ann_bin = 0
  pool_bin = 0
  pq_bin = 0
  q_bin = 0
  do i = 1, size(bins)
    bins(i) = (i - 1) * binwidth
  end do

  open(file=rates_file, unit=20)
  read(20, *) base_rates
  close(20)
  write(*, '(a)', advance='no') "Progress: ["

  allocate(is_quencher(n_sites))
  is_quencher = .false.
  n_quenchers = int(n_sites * rho_q)
  allocate(quenchers(n_quenchers))
  
  call cpu_time(start_time)
  open(file=loss_file, unit=20)
  i = 0
  ! keep iterating till we get a decent number of counts
  ! pool decays and pre-quencher decays are emissive, so
  ! those are what we're concerned with for the fit
  do while ((maxval(pool_bin) + maxval(pq_bin)).lt.max_count)
    seed = i
    call random_seed(put=seed)
    is_quencher = .false.
    call allocate_quenchers(n_quenchers, is_quencher, quenchers)
    call fix_base_rates()
    if (mod(i, 100).eq.0) then
      write(*, *) i, maxval(ann_bin), maxval(pool_bin), maxval(pq_bin), maxval(q_bin)
    end if
    n_i = 0
    n_pq = 0
    n_q = 0
    n_current = 0
    t = 0.0_dp
    do while (t.lt.t_pulse)
      ! ensure we generate excitons, otherwise
      ! n_current = 0 and we never start
      call mc_step(dt, n_quenchers)
    end do
    do while ((n_current.gt.0).and.(t.lt.max_time))
      call mc_step(dt, n_quenchers)
    end do
    i = i + 1
  end do

  write(*, *) "]."
  close(20)

  open(file=counts_file, unit=20)
  do j = 1, size(bins)
    write(20, '(F8.3, 4I8)') bins(j), ann_bin(j), pool_bin(j),&
      pq_bin(j), q_bin(j)
  end do
  close(20)

  call cpu_time(end_time)
  write(*, *) "Time elapsed: ", end_time - start_time
  write(*, *) "Number of iterations: ", i

  deallocate(is_quencher)
  deallocate(quenchers)
  deallocate(seed)
  deallocate(n_i)
  deallocate(n_pq)
  deallocate(n_q)
  deallocate(rates)
  deallocate(base_rates)
  deallocate(neighbours)
  deallocate(neighbours_temp)
  deallocate(pulse)
  deallocate(bins)
  deallocate(ann_bin)
  deallocate(pool_bin)
  deallocate(pq_bin)
  deallocate(q_bin)

  contains

    function randint(upper) result(i)
      ! returns an integer between 1 and number for array indexing
      implicit none
      integer, intent(in) :: upper
      integer(ip) :: i
      real :: r
      call random_number(r)
      ! random_number can return 0; stop that
      do while (r.eq.0.0_dp)
        call random_number(r)
      end do
      i = ceiling(upper * r)
      if (i.eq.0) then
        write(*, *) "randint returned 0!", upper, r, upper * r
      end if
    end function randint

    subroutine allocate_quenchers(n_quenchers, is_quencher, quenchers)
      ! set n_quenchers trimers as quenchers randomly
      implicit none
      logical(C_bool), dimension(:) :: is_quencher
      integer(ip), dimension(:) :: quenchers
      integer :: n_quenchers, i, choice
      real(dp) :: rho_q, pq_hop
      do i = 1, n_quenchers
         choice = randint(size(quenchers))
         do while (is_quencher(choice))
           choice = randint(size(quenchers))
         end do
         is_quencher(choice) = .true.
         quenchers(i) = choice
      end do
    end subroutine allocate_quenchers

    subroutine fix_base_rates()
      ! the python code outputs general rates for each trimer,
      ! based on its neighbours etc. the hopping rate to the
      ! pre-quencher is also in there, but only trimers that
      ! are quenchers should have this, so zero it if necessary
      implicit none
      integer :: i
      do i = 1, n_sites
        if (is_quencher(i).eqv..false.) then
          base_rates((i * rate_size) - 2) = 0.0_dp 
        end if
      end do
    end subroutine fix_base_rates

    function get_rates(ind, t, rate_type) result(rates)
      ! return a set of rates depending on which trimer
      ! we're on and whether it's a quencher etc.
      implicit none
      real(dp) :: rates(rate_size)
      character(4) :: rate_type
      integer(ip) :: ind, start, end_, n, t_index, k, n_mult
      real(dp) :: t, ft, xsec, sigma_ratio, n_pigments, ann_fac
      if (trim(rate_type).eq."PQ") then
        start = ((n_sites) * rate_size) + 1
        end_ = start + rate_size - 1
        n = n_pq(ind)
      else if (trim(rate_type).eq."Q") then
        start = ((n_sites + 1) * rate_size) + 1
        end_ = start + rate_size - 1
        n = n_q(ind)
      else if (rate_type.eq."POOL") then
        start = ((ind - 1) * rate_size) + 1
        end_ = start + rate_size - 1
        n = n_i(ind)
      else
        write (*, *) "get_rates - rate_type error"
      end if
      rates = base_rates(start:end_)
      if ((start.lt.0).or.(end_.gt.size(base_rates))) then
        write (*,*) "get_rates bounds error: start ",&
                    start, " end, ", end_
      end if
      sigma_ratio = 1.5_dp
      n_pigments = 24.0_dp
      if (t < t_pulse) then
        t_index = int(t / dt) + 1
        ft = pulse(t_index)
        xsec = 1.1E-14
        if (((1 + sigma_ratio) * n).le.n_pigments) then
          rates(1) = xsec * fluence * ft * &
            ((n_pigments - (1 + sigma_ratio) * n)/ n_pigments)
        end if
      else
        rates(1) = 0.0_dp
      end if
      do k = 2, rate_size - 1
        rates(k) = rates(k) * n
      end do
      ann_fac = (n * (n - 1)) / 2.0_dp
      rates(rate_size) = rates(rate_size) * ann_fac
    end function get_rates

    subroutine move(ind, process, pop_loss, rate_type)
      implicit none
      character(4) :: rate_type
      integer(ip), intent(in) :: ind, process
      integer(ip) :: k, nn, choice, n_mult
      logical(C_Bool), intent(inout) :: pop_loss(4)

      if (rate_type.eq."POOL") then
        ! normal trimer
        if (process.eq.1) then
          ! generation
          n_i(ind)  = n_i(ind)  + 1
          n_current = n_current + 1
        else if ((process.gt.1).and.(process.lt.(rate_size - 2))) then
          ! hop to neighbour
          nn = neighbours(ind, process)
          n_i(ind) = n_i(ind) - 1
          n_i(nn)  = n_i(nn)  + 1
        else if (process.eq.rate_size - 2) then
          ! hop to pre-quencher
          n_i(ind)  = n_i(ind)  - 1
          n_pq(ind) = n_pq(ind) + 1
        else if (process.eq.rate_size - 1) then
          ! decay
          n_i(ind)  = n_i(ind)  - 1
          n_current = n_current - 1
          pop_loss(2) = .true.
        else if (process.eq.rate_size) then
          ! annihilation
          n_i(ind)  = n_i(ind)  - 1
          n_current = n_current - 1
          pop_loss(1) = .true.
        else
          write(*, *) "Move function failed on trimer.", ind, process
        end if
      else if (trim(rate_type).eq."PQ") then
        ! pre-quencher
        if (process.eq.1) then
          ! generation
          n_pq(ind) = n_pq(ind) + 1
          n_current = n_current + 1
        else if (process.eq.(rate_size - 3)) then
          ! hop back to pool trimer
          n_pq(ind) = n_pq(ind) - 1
          n_i(ind)  = n_i(ind)  + 1
        else if (process.eq.rate_size - 2) then
          ! hop to quencher
          n_pq(ind) = n_pq(ind) - 1
          n_q(ind)  = n_q(ind)  + 1
        else if (process.eq.rate_size - 1) then
          ! pq decay
          n_pq(ind) = n_pq(ind) - 1
          n_current = n_current - 1
          pop_loss(3) = .true.
        else if (process.eq.rate_size) then
          ! annihilation
          n_pq(ind) = n_pq(ind) - 1
          n_current = n_current - 1
          pop_loss(1) = .true.
        else
          write(*, *) "Move function failed on pre-quencher."
        end if
      else if (trim(rate_type).eq."Q") then
        ! quencher
        if (process.eq.0) then
          ! generation
          n_q(ind)  = n_q(ind)  + 1
          n_current = n_current + 1
        else if (process.eq.(rate_size - 2)) then
          ! hop back to pre-quencher
          n_q(ind)  = n_q(ind)  - 1
          n_pq(ind) = n_pq(ind) + 1
        else if (process.eq.rate_size - 1) then
          ! q decay
          n_q(ind)  = n_q(ind)  - 1
          n_current = n_current - 1
          pop_loss(4) = .true.
        else if (process.eq.rate_size) then
          ! annihilation
          n_q(ind)  = n_q(ind)  - 1
          n_current = n_current - 1
          pop_loss(1) = .true.
        end if
      end if
      if ((n_i(ind).lt.0).or.(n_pq(ind).lt.0).or.(n_q(ind).lt.0)) then
        write(*, *) "move set occupancy to < 0!!!", n_i(ind), n_pq(ind), n_q(ind)
        stop
      end if
    end subroutine move

    subroutine mc_step(dt, n_quenchers)
      implicit none
      character(4) :: rate_type
      integer(ip) :: n_attempts, nonzero, choice, i, j, k, n_quenchers, ri, trimer
      logical(C_Bool) :: pop_loss(4)
      real(dp) :: dt, rho_q, rand
      real(dp), dimension(rate_size) :: rates, probs

      n_attempts = n_sites + (2 * n_quenchers)
      ! attempt to do something on each site
      ! (including quenchers), on average
      do i = 1, n_attempts
        pop_loss = (/.false., .false., .false., .false./)
        ri = randint(n_attempts)
        if (ri.le.n_sites) then
          trimer = ri
          rate_type = "POOL"
        else if ((ri.gt.n_sites).and.&
          (ri.le.(n_sites + n_quenchers))) then
          trimer = quenchers(ri - n_sites)
          rate_type = "PQ"
        else if (ri.gt.(n_sites + n_quenchers)) then
          trimer = quenchers(ri - (n_sites + n_quenchers))
          rate_type = "Q"
        end if

        rates = get_rates(trimer, t, rate_type)
        ! no point bothering with this if no moves are possible
        if (.not.any(rates.gt.0.0_dp)) then
          cycle
        end if
        ! nonzero is the number of possible moves
        nonzero = 0
        do j = 1, rate_size
          if (rates(j).gt.0.0_dp) then
            nonzero = nonzero + 1
          end if
        end do

        ! attempt each possible move once, on average
        do j = 1, nonzero
          ! check whether any moves are possible now
          if (.not.any(rates.gt.0.0_dp)) then
            exit
          end if
          probs = [(rates(k) * exp(-1.0_dp * rates(k) * dt), &
            k = 1, rate_size)]
          ! pick a possible move
          choice = randint(nonzero)
          ri = 0
          do k = 1, rate_size
            if (probs(k).gt.0.0_dp) then
              ri = ri + 1
            end if
            if (ri.eq.choice) then
              choice = k
              exit
            end if
          end do
          ! monte carlo test
          call random_number(rand)
          if (rand.lt.probs(choice)) then
            ! successful move
            call move(trimer, choice, pop_loss, rate_type)
            ! we need to recalculate the rates (population changed)
            rates = get_rates(trimer, t, rate_type)
          end if
        end do

        if (any(pop_loss)) then
          if (pop_loss(1)) then
            ann_bin(floor(t / binwidth) + 1) = &
              ann_bin(floor(t / binwidth) + 1) + 1
          else if (pop_loss(2)) then
            pool_bin(floor(t / binwidth) + 1) = &
              pool_bin(floor(t / binwidth) + 1) + 1
          else if (pop_loss(3)) then
            pq_bin(floor(t / binwidth) + 1) = &
              pq_bin(floor(t / binwidth) + 1) + 1
          else if (pop_loss(4)) then
            q_bin(floor(t / binwidth) + 1) = &
              q_bin(floor(t / binwidth) + 1) + 1
          end if
          ! the following just outputs all decay times and types
          ! to a file for the python to then interpret
          do k = 1, size(pop_loss)
            if (pop_loss(k)) then
              ! append loss time and decay type to file here!
              write(20, '(F10.4, a, I1)') t, " ", k
            end if
          end do
        end if
      end do
      t = t + dt
    end subroutine mc_step

end program iteration
