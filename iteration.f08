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
  logical(c_bool), dimension(:), allocatable :: quenchers
  integer(ip) :: i, j, n_iter, n_sites, max_neighbours, rate_size,&
    n_current, n_pq, n_q, seed_size, stat, col, max_count
  integer(ip), dimension(:), allocatable :: n_i, pq, q, c_pq, c_q, seed
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
  rate_size = max_neighbours + 4
  write(*, *) "n_sites = ", n_sites
  write(*, *) "max neighbours = ", max_neighbours
  write(*, *) "rho_q = ", rho_q
  write(*, *) "fluence = ", fluence
  write(*, *) "mu = ", mu
  write(*, *) "fwhm = ", fwhm
  write(*, *) "binwidth = ", binwidth
  write(*, *) "max count = ", max_count
  write(*, *) "rates file = ", rates_file
  write(*, *) "neighbours file = ", neighbours_file

  i = scan(rates_file, "/\", .true.)
  file_path = rates_file(:i)
  write(prefix_long, '(I4, a, F4.2, a, ES8.2, a)') n_iter, "_", rho_q,&
    "_", fluence, "_"
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
  allocate(base_rates(n_sites * rate_size))
  allocate(rates(n_sites * rate_size))
  allocate(n_i(n_sites))
  ! need a check for these arrays getting full
  ! something with maxlen(pq/q)
  allocate(pq(int(1.1e-14 * fluence * n_sites)))
  allocate(q(int(1.1e-14 * fluence * n_sites)))
  allocate(c_pq(int(1.1e-14 * fluence * n_sites)))
  allocate(c_q(int(1.1e-14 * fluence * n_sites)))
  pq = 0
  q = 0
  c_pq = 0
  c_q = 0
  allocate(n_gen(n_iter))
  allocate(n_ann(n_iter))
  allocate(n_po_d(n_iter))
  allocate(n_pq_d(n_iter))
  allocate(n_q_d(n_iter))
  n_gen = 0
  n_ann = 0
  n_po_d = 0
  n_pq_d = 0
  n_q_d = 0

  dt = 1.0_dp
  t_pulse = 2.0_dp * mu
  allocate(pulse(int(t_pulse / dt)))
  write(*, *) t_pulse, mu
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

  allocate(quenchers(n_sites))
  quenchers = .false.
  
  call cpu_time(start_time)
  open(file=loss_file, unit=20)
  i = 0
  ! keep iterating till we get a decent number of counts
  do while ((maxval(pool_bin) < max_count).and.(maxval(pq_bin) < max_count)&
       .and.(maxval(q_bin) < max_count))
    seed = i
    call random_seed(put=seed)
    call allocate_quenchers(quenchers, rho_q)
    call fix_base_rates()
    if (mod(i, 100).eq.0) then
      write(*, *) i, maxval(ann_bin), maxval(pool_bin), maxval(pq_bin), maxval(q_bin)
    end if
    n_i = 0
    n_current = 0
    t = 0.0_dp
    do j = 1, n_sites
      call update_rates(j, n_i(j), t)
    end do
    do while (t < t_pulse)
      call mc_step(dt, rho_q, i)
    end do
    do j = 1, n_sites
      call update_rates(j, n_i(j), t)
    end do
    do while ((n_current > 0).and.(t < max_time))
      call mc_step(dt, rho_q, i)
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

  deallocate(quenchers)
  deallocate(seed)
  deallocate(pq)
  deallocate(q)
  deallocate(c_pq)
  deallocate(c_q)
  deallocate(n_i)
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

    subroutine allocate_quenchers(quenchers, rho_q)
      implicit none
      logical(C_bool), dimension(:) :: quenchers
      integer :: n_q, i, choice
      real(dp) :: rho_q, pq_hop
      n_q = int(size(quenchers) * rho_q)
      do i = 1, n_q
         choice = randint(size(quenchers))
         do while (quenchers(choice))
           choice = randint(size(quenchers))
         end do
         quenchers(choice) = .true.
      end do
    end subroutine allocate_quenchers

    subroutine fix_base_rates()
      implicit none
      integer :: i
      do i = 1, n_sites
        if (quenchers(i).eqv..false.) then
          base_rates((i * rate_size) - 2) = 0.0_dp 
        end if
      end do
    end subroutine fix_base_rates

    subroutine count_multiples(array, c, n_mult)
      ! return list of counts of elements that are > 1
      ! needs testing! after running, the array c should
      ! contain counts of each element, with counts > 1
      ! located where the first occurrence of that element
      ! is in arrayexec format error fortran
      implicit none
      integer(ip), dimension(:), intent(in) :: array
      integer, dimension(:) :: c
      integer :: i, j, n_mult
      c = 0
      n_mult = 0
      do i = 1, size(array)
        if (array(i).le.0) then
          cycle
        else
          do j = 1, i
            if (array(j).eq.array(i)) then
              c(i) = c(i) + 1
            end if
          end do
        end if
      end do
      do i = 1, size(array)
        if (c(i).gt.1) then
          n_mult = n_mult + 1
        end if
      end do
    end subroutine count_multiples

    subroutine bkl(rand, ind, process, k_tot)
      implicit none
      real(dp), intent(in) :: rand
      real(dp) :: k_tot
      real(dp), dimension(size(rates)) :: c_rates
      integer(ip) :: l, r, m, k, ind, process
      ! update cumulative rates
      c_rates = [(sum(rates(1:k)), k = 1, size(rates))]
      ! we need this to update the time in kinetic Monte Carlo
      k_tot = c_rates(size(rates))
      ! binary search to find correct process
      l = 1
      r = size(c_rates)
      do while (l < r) 
        m = (l + r) / 2
        if (c_rates(m) < (rand * k_tot)) then
          l = m + 1
        else
          r = m
        end if
      end do
      ! rates has one long index representing every possible process
      ! but to figure out how to update the system we need to know
      ! which trimer we're on and which process we've picked
      ind = ((l - 1) / rate_size) + 1
      process = mod(l - 1, rate_size) + 1
      ! write(*, *) "bkl: l = ", l, ", ", rates(l - 1), rates(l), "ind = ", ind, "process = ",&
      !   process, "k_tot = ", k_tot
      ! write(*, *) c_rates
    end subroutine bkl

    subroutine update_rates(ind, n, t)
      implicit none
      integer(ip) :: ind, start, end_, n, t_index, k, n_mult
      real(dp) :: t, ft, xsec, sigma_ratio, n_pigments, ann_fac
      ! fortran slicing is inclusive on both ends and it's 1-based
      ! check this is correct!
      start = ((ind - 1) * rate_size) + 1
      end_ = start + rate_size - 1
      rates(start:end_) = base_rates(start:end_)
      if ((start.lt.0).or.(end_.gt.size(rates))) then
        write (*,*) "update_rates: start ", start, " end, ", end_
      end if
      ! write(*, *) "update_rates before: i = ", ind, "n = ", n,&
      !   "rates = ", rates(start:start + rate_size)
      sigma_ratio = 1.5_dp
      n_pigments = 24.0_dp
      if (t < t_pulse) then
        t_index = int(t / dt) + 1
        ft = pulse(t_index)
        if ((t_index.le.0).or.(t_index.gt.size(pulse))) then
          ! write(*, *) dt, t, int(t / dt) + 1, size(pulse), ft
        end if
        xsec = 1.1E-14
        if ((1 + sigma_ratio) * n <= n_pigments) then
          rates(start) = xsec * fluence * ft * &
            ((n_pigments - (1 + sigma_ratio) * n)/ n_pigments)
        end if
      else
        rates(start) = 0.0_dp
      end if
      do k = start + 1, end_ - 1
        rates(k) = rates(k) * n
        ! rates(k) = 0.0_dp
      end do
      if (mod(ind, rate_size) == n_sites - 1) then
        ! pre-quencher
        ! this is the hard bit in fortran tbh
        call count_multiples(pq, c_pq, n_mult)
        ann_fac = 0.
        do k = 1, size(c_pq)
          if (c_pq(k).gt.1) then
            ann_fac = ann_fac + (c_pq(k) * (c_pq(k) - 1) / 2.0_dp)
          end if
        end do
        rates(end_) = rates(end_) * ann_fac
      else if (mod(ind, rate_size) == n_sites) then
        ! quencher
        ann_fac = 0.
        call count_multiples(q, c_q, n_mult)
        do k = 1, size(c_q)
          if (c_q(k).gt.1) then
            ann_fac = ann_fac + (c_q(k) * (c_q(k) - 1) / 2.0_dp)
          end if
        end do
        rates(end_) = rates(end_) * ann_fac
      else
        ann_fac = (n * (n - 1)) / 2.0_dp
        ! ann_fac = 0.0_dp
        rates(end_) = rates(end_) * ann_fac
      end if
    end subroutine update_rates

    subroutine move(ind, process, pop_loss, iter)
      implicit none
      integer(ip), intent(in) :: ind, process, iter
      integer(ip) :: k, nn, choice, n_mult
      logical(C_Bool), intent(inout) :: pop_loss(4)
      ! pq and q should be set to the same size, roughly
      ! xsec * fluence? i guess? there's gonna have to be some
      ! check for n_pq/n_q and resizing
      if (ind.lt.(n_sites - 1)) then
        ! normal trimer
        if (process.eq.1) then
          ! generation
          n_i(ind) = n_i(ind) + 1
          n_current = n_current + 1
          ! n_gen(iter) = n_gen(iter) + 1
          call update_rates(ind, n_i(ind), t)
          ! write(*, *) "generation on ", ind, ". n_current = ", n_current
        else if ((process.gt.1).and.(process.lt.(rate_size - 2))) then
          ! hop to neighbour
          nn = neighbours(ind, process)
          n_i(ind) = n_i(ind) - 1
          n_i(nn) = n_i(nn) + 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(nn, n_i(nn), t)
          ! write(*, *) "hop from ", ind, " to ", nn
        else if (process == rate_size - 2) then
          ! hop to pre-quencher
          n_i(ind) = n_i(ind) - 1
          n_i(n_sites - 1) = n_i(n_sites - 1) + 1
          pq(n_pq + 1) = ind
          n_pq = n_pq + 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(n_sites - 1, n_i(n_sites - 1), t)
          ! write(*, *) "hop from ", ind, " to pq"
        else if (process == rate_size - 1) then
          ! decay
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          call update_rates(ind, n_i(ind), t)
          pop_loss(2) = .true.
          ! n_po_d(iter) = n_po_d(iter) + 1
          ! write(*, *) "decay on ", ind, ". n_current = ", n_current
        else if (process == rate_size) then
          ! annihilation
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          call update_rates(ind, n_i(ind), t)
          pop_loss(1) = .true.
          ! n_ann(iter) = n_ann(iter) + 1
          ! write(*, *) "ann on ", ind, ". n_current = ", n_current
        else
          write(*, *) "Move function failed on trimer.", ind, process
        end if
      else if (ind.eq.(n_sites - 1)) then
        ! pre-quencher
        if (process == 0) then
          ! generation
          n_i(ind) = n_i(ind) + 1
          n_current = n_current + 1
          ! n_gen(i) = n_gen(i) + 1
          ! need to generate a trimer for it to hop to
          choice = randint(n_sites - 2)
          pq(n_pq + 1) = choice
          n_pq = n_pq + 1
          call update_rates(ind, n_i(ind), t)
          ! write(*, *) "generation on pq. n_current = ", n_current,&
          !   "n_pq = ", n_pq, "prev = ", choice
        else if (process.eq.(rate_size - 3)) then
          ! hop back to pool trimer
          ! pick one at random
          ! think about what order this all needs to be done in
          n_i(ind) = n_i(ind) - 1
          choice = randint(n_pq)
          n_i(pq(choice)) = n_i(pq(choice)) + 1
          ! write(*, *) "hop from pq to ", pq(choice), "n_pq = ", n_pq - 1
          call update_rates(pq(choice), n_i(pq(choice)), t)
          pq(choice) = pq(n_pq)
          pq(n_pq) = -1
          n_pq = n_pq - 1
          call update_rates(ind, n_i(ind), t)
        else if (process == rate_size - 2) then
          ! hop to quencher
          n_i(ind) = n_i(ind) - 1
          n_i(n_sites) = n_i(n_sites) + 1
          choice = randint(n_pq)
          q(n_q + 1) = pq(choice)
          pq(choice) = pq(n_pq)
          pq(n_pq) = -1
          n_pq = n_pq - 1
          n_q = n_q + 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(n_sites, n_i(n_sites), t)
          ! write(*, *) "hop from pq to q. n_pq = ", n_pq, " n_q = ", n_q
        else if (process == rate_size - 1) then
          ! decay
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          choice = randint(n_pq)
          pq(choice) = pq(n_pq)
          pq(n_pq) = -1
          n_pq = n_pq - 1
          call update_rates(ind, n_i(ind), t)
          pop_loss(3) = .true.
          ! n_pq_d(iter) = n_pq_d(iter) + 1
          ! write(*, *) "decay on ", ind, ". n_current = ",&
          !   n_current, " n_pq = ", n_pq
        else if (process == rate_size) then
          ! annihilation
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          ! if it's an annihilation it must be a multiple
          ! find the multiples
          call count_multiples(pq, c_pq, n_mult)
          nn = 0
          do k = 1, size(c_pq)
            if (c_pq(k) > 1) then
              nn = nn + 1
            end if
          end do
          ! pick one of the multiples at random
          choice = randint(nn)
          nn = 0
          do k = 1, size(c_pq)
            if (c_pq(k) > 1) then
              nn = nn + 1
            end if
            if (nn.eq.choice) then
              ! get the index in pq of the first
              ! occurence of this multiple
              choice = k
              exit
            end if
          end do
          ! now get rid of it
          pq(choice) = pq(n_pq)
          pq(n_pq) = -1
          n_pq = n_pq - 1
          pop_loss(1) = .true.
          call update_rates(ind, n_i(ind), t)
          ! n_ann(iter) = n_ann(iter) + 1
          ! write(*, *) "ann on ", ind, ". n_current = ",&
          !   n_current, " n_pq = ", n_pq
        else
          write(*, *) "Move function failed on pre-quencher."
        end if
      else if (ind.eq.(n_sites)) then
        ! quencher
        if (process == 0) then
          ! generation
          n_i(ind) = n_i(ind) + 1
          n_current = n_current + 1
          ! n_gen(i) = n_gen(i) + 1
          ! need to generate a trimer for it to hop to
          choice = randint(n_sites - 2)
          q(n_q + 1) = choice
          n_q = n_q + 1
          call update_rates(ind, n_i(ind), t)
          ! write(*, *) "generation on q. n_current = ",&
          !   n_current, "n_pq = ", n_q, "prev = ", choice
        else if (process.eq.(rate_size - 2)) then
          ! hop back to pre-quencher
          n_i(ind) = n_i(ind) - 1
          choice = randint(n_q)
          pq(n_pq + 1) = q(choice)
          q(choice) = q(n_pq)
          q(n_q) = -1
          n_q = n_q - 1
          n_pq = n_pq + 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(n_sites - 1, n_i(n_sites - 1), t)
          ! write(*, *) "hop from q to pq. n_q = ", n_q, " n_pq = ", n_pq
        else if (process == rate_size - 1) then
          ! decay
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          choice = randint(n_q)
          q(choice) = q(n_q)
          q(n_q) = -1
          n_q = n_q - 1
          call update_rates(ind, n_i(ind), t)
          pop_loss(3) = .true.
          ! n_q_d(iter) = n_q_d(iter) + 1
          ! write(*, *) "decay on ", ind, ". n_current = ",&
          !   n_current, " n_q = ", n_q
        else if (process == rate_size) then
          ! annihilation
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          call count_multiples(q, c_q, n_mult)
          nn = 0
          do k = 1, size(c_q)
            if (c_q(k) > 1) then
              nn = nn + 1
            end if
          end do
          choice = randint(nn)
          nn = 0
          do k = 1, size(c_q)
            if (c_q(k) > 1) then
              nn = nn + 1
            end if
            if (nn.eq.choice) then
              choice = k
              exit
            end if
          end do
          q(choice) = q(n_q)
          q(n_pq) = -1
          n_q = n_q - 1
          pop_loss(1) = .true.
          call update_rates(ind, n_i(ind), t)
          ! n_ann(iter) = n_ann(iter) + 1
          ! write(*, *) "ann on ", ind, ". n_current = ",&
          !   n_current, " n_q = ", n_q
        end if
      end if

    end subroutine move

    subroutine mc_step(dt, rho_q, iter)
      implicit none
      integer(ip) :: n_attempts, i, k,&
                     trimer, nonzero, choice
      integer(ip), intent(in) :: iter
      logical(C_Bool) :: pop_loss(4)
      real(dp) :: dt, rho_q, rand
      real(dp), dimension(rate_size) :: probs
      if (rho_q.gt.0.0) then
        n_attempts = n_sites
      else
        n_attempts = n_sites - 2
      end if
      do i = 1, n_attempts
        pop_loss = (/.false., .false., .false., .false./)
        trimer = randint(n_attempts)
        call update_rates(trimer, n_i(trimer), t)
        ! need to check this is right lol
        probs = [(rates(k) * exp(-1.0_dp * rates(k) * dt), &
          k = ((trimer - 1) * rate_size) + 1, (trimer * rate_size))]
        do k = 1, size(probs)
          choice = randint(size(probs))
          if (probs(choice).gt.0.0_dp) then
            call random_number(rand)
            if (rand.lt.probs(choice)) then
              call move(trimer, choice, pop_loss, iter)
            end if
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

    function kmc_step(iter) result(res)
      implicit none
      real(dp) :: r1, r2, k_tot
      integer(ip), intent(in) :: iter
      integer(ip) :: k, l, i, process, res
      logical(C_Bool) :: pop_loss(4)
      if ((n_current.eq.0).and.(t.gt.t_pulse)) then
        ! do nothing
        res = -1
        return
      else
        pop_loss = .false.
        call random_number(r1)
        call random_number(r2)
        call bkl(r1, i, process, k_tot)
        if (process.eq.0) then
          write(*, *) "KMC - process = 0"
        end if
        call move(i, process, pop_loss, iter)
        t = t - (1.0_dp / k_tot) * log(r2)
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
          do k = 1, size(pop_loss)
            if (pop_loss(k)) then
              write(20, '(F10.4, a, I1)') t, " ", k
            end if
          end do
        end if
      end if
      res = 0
    end function kmc_step

end program iteration
