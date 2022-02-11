program iteration
  use iso_fortran_env
  use iso_c_binding
  implicit none
  integer, parameter :: dp = c_double
  integer, parameter :: short = c_short
  integer, parameter :: ip = c_int
  real, parameter :: pi = 3.1415926535

  character(len=200) :: params_file, rates_file, neighbours_file,&
    loss_file, seed_str
  logical(c_bool), dimension(:), allocatable :: quenchers
  integer(ip) :: i, n_iter, n_sites, rate_size,&
    n_current, n_pq, n_q, seed, stat
  integer(ip), dimension(:), allocatable :: n_i, pq, q
  integer(ip), dimension(:), allocatable :: neighbours_temp
  integer(ip), dimension(:, :), allocatable :: neighbours
  real(dp) :: mu, fluence, t, dt, rho_q, sigma, fwhm
  real(dp), dimension(:), allocatable :: rates, base_rates, pulse

  call get_command_argument(1, params_file)
  stat = 0
  dt = 0.1_dp
  open(file=params_file, unit=20)
  read(20, *) n_sites
  read(20, *) rate_size
  read(20, *) rho_q
  read(20, *) fluence
  read(20, *) rates_file
  read(20, *) neighbours_file
  close(20)

  allocate(rates(n_sites * rate_size))
  allocate(neighbours_temp(n_sites * rate_size))
  allocate(neighbours(n_sites, rate_size))
  allocate(n_i(n_sites))
  ! need a check for this array getting full
  ! something with maxlen(pq/q)
  allocate(pq(int(1.1e-14 * fluence * n_sites)))
  allocate(q(int(1.1e-14 * fluence * n_sites)))
  allocate(pulse(200))
  fwhm = 50.0_dp
  sigma = fwhm / (2.0_dp * (sqrt(2.0_dp * log(2.0_dp))))
  do i = 1, 200
    pulse(i) = 1.0_dp / (sigma * sqrt(2.0_dp * pi)) * &
      exp(-1.0_dp * (float(i) - mu)**2 / (sqrt(2.0_dp) * sigma)**2)
  end do

  open(file=rates_file, unit=20)
  read(20, *) rates
  close(20)
  open(file=neighbours_file, unit=20)
  read(20, *) neighbours_temp
  close(20)
  neighbours = reshape(neighbours_temp, (/n_sites, rate_size/))
  
  open(file=loss_file, unit=20)
  do i = 1, n_iter
    call random_seed(i)
    call allocate_quenchers(quenchers, rho_q)
    t = 0.0_dp
    do while (t < 200.0_dp)
      call mc_step(dt, rho_q)
    end do
    do while (stat.eq.0)
      stat = kmc_step()
    end do
    ! idea - could generate an array of bins and bin the decays as we
    ! go. then reduce this if we're parallelising
  end do
  close(20)
  stop

  contains

    integer(ip) function randint(upper) result(i)
      ! returns an integer between 1 and number for array indexing
      implicit none
      integer, intent(in) :: upper
      real :: r
      call random_number(r)
      i = ceiling(upper * r)
    end function randint

    subroutine allocate_quenchers(quenchers, rho_q) bind(C)
      implicit none
      logical(C_bool), dimension(:) :: quenchers
      integer :: n_q, i, choice
      real(dp) :: rho_q
      n_q = int(size(quenchers) * rho_q)
      do i = 1, n_q
         choice = randint(size(quenchers))
         do while (quenchers(choice))
           choice = randint(size(quenchers))
         end do
         quenchers(choice) = .true.
      end do
    end subroutine allocate_quenchers

    subroutine count_multiples(array, c, n_mult) bind(C)
      ! return list of counts of elements that are > 1
      ! needs testing! after running, the array c should
      ! contain counts of each element, with counts > 1
      ! located where the first occurrence of that element
      ! is in array
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

    subroutine bkl(rand, ind, process, k_tot) bind(C)
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
      l = 0
      r = size(c_rates)
      do while (l < r) 
        m = (l + r) / 2
        if (c_rates(m) < rand * k_tot) then
          l = m + 1
        else
          r = m
        end if
      end do
      ! rates has one long index representing every possible process
      ! but to figure out how to update the system we need to know
      ! which trimer we're on and which process we've picked
      ind = (l / rate_size) + 1
      process = mod(l, rate_size) + 1
    end subroutine bkl

    subroutine update_rates(ind, n, t) bind(C)
      implicit none
      integer(ip) :: ind, n, t_index, k
      real(dp) :: t, ft, xsec, sigma_ratio, n_pigments
      ! fortran slicing is inclusive on both ends and it's 1-based
      ! check this is correct!
      rates(ind:ind + rate_size) = base_rates(ind:ind + rate_size)
      if (t < 2.0_dp * mu) then
        t_index = int(t)
        ft = pulse(t_index)
        xsec = 1.1E-14
        sigma_ratio = 3.0_dp
        n_pigments = 24.0_dp
        if ((1 + sigma_ratio) * n <= n_pigments) then
          rates(ind + 1) = xsec * fluence * ft * &
            ((n_pigments - (1 + sigma_ratio) * n)/ n_pigments)
        end if
      end if
      do k = ind, ind + rate_size - 1
        rates(k) = rates(k) * n
      end do
      if (mod(ind, rate_size) == n_sites - 1) then
        ! pre-quencher
        ! this is the hard bit in fortran tbh
      else if (mod(ind, rate_size) == n_sites) then
        ! quencher
      else
        rates(ind + rate_size) = rates(ind + rate_size) &
                               * (n * (n - 1) / 2.0_dp)
      end if
    end subroutine update_rates

    subroutine move(ind, process, pop_loss) bind(C)
      implicit none
      integer(ip), intent(in) :: ind, process
      integer(ip) :: k, nn, choice, n_mult
      integer(ip), dimension(:), allocatable :: c
      logical(C_Bool), intent(inout) :: pop_loss(4)
      ! pq and q should be set to the same size, roughly
      ! xsec * fluence? i guess? there's gonna have to be some
      ! check for n_pq/n_q and resizing
      allocate(c(size(pq)))
      if (ind < n_sites - 1) then
        ! normal trimer
        if (process == 0) then
          ! generation
          n_i(ind) = n_i(ind) + 1
          n_current = n_current + 1
          call update_rates(ind, n_i(ind), t)
        else if ((process.gt.0).and.(process.lt.(rate_size - 2))) then
          ! hop to neighbour
          nn = neighbours(ind, process)
          n_i(ind) = n_i(ind) - 1
          n_i(nn) = n_i(nn) + 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(nn, n_i(nn), t)
        else if (process == rate_size - 2) then
          ! hop to pre-quencher
          n_i(ind) = n_i(ind) - 1
          n_i(n_sites - 1) = n_i(n_sites - 1) + 1
          pq(n_pq + 1) = ind
          n_pq = n_pq + 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(n_sites - 1, n_i(n_sites - 1), t)
        else if (process == rate_size - 1) then
          ! decay
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          call update_rates(ind, n_i(ind), t)
          pop_loss(2) = .true.
        else if (process == rate_size) then
          ! annihilation
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          call update_rates(ind, n_i(ind), t)
          pop_loss(1) = .true.
        else
          write(*, *) "Move function failed on trimer."
        end if
      else if (ind.eq.(n_sites - 1)) then
        ! pre-quencher
        if (process == 0) then
          ! generation
          n_i(ind) = n_i(ind) + 1
          n_current = n_current + 1
          ! need to generate a trimer for it to hop to
          choice = randint(n_sites - 2)
          pq(n_pq + 1) = choice
          n_pq = n_pq + 1
          call update_rates(ind, n_i(ind), t)
        else if (process.eq.(rate_size - 3)) then
          ! hop back to pool trimer
          ! pick one at random
          ! think about what order this all needs to be done in
          n_i(ind) = n_i(ind) - 1
          choice = randint(n_pq)
          n_i(pq(choice)) = n_i(pq(choice)) + 1
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
        else if (process == rate_size) then
          ! annihilation
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          ! if it's an annihilation it must be a multiple
          ! find the multiples
          call count_multiples(pq, c, n_mult)
          nn = 0
          do k = 1, size(c)
            if (c(k) > 1) then
              nn = nn + 1
            end if
          end do
          ! pick one of the multiples at random
          choice = randint(nn)
          nn = 0
          do k = 1, size(c)
            if (c(k) > 1) then
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
        else
          write(*, *) "Move function failed on pre-quencher."
        end if
      else if (ind.eq.(n_sites)) then
        ! quencher
        if (process == 0) then
          ! generation
          n_i(ind) = n_i(ind) + 1
          n_current = n_current + 1
          ! need to generate a trimer for it to hop to
          choice = randint(n_sites - 2)
          q(n_q + 1) = choice
          n_q = n_q + 1
          call update_rates(ind, n_i(ind), t)
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
        else if (process == rate_size) then
          ! annihilation
          n_i(ind) = n_i(ind) - 1
          n_current = n_current - 1
          call count_multiples(q, c, n_mult)
          nn = 0
          do k = 1, size(c)
            if (c(k) > 1) then
              nn = nn + 1
            end if
          end do
          choice = randint(nn)
          nn = 0
          do k = 1, size(c)
            if (c(k) > 1) then
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
        end if
      end if

    end subroutine move

    subroutine mc_step(dt, rho_q) bind(C)
      implicit none
      integer(ip) :: n_attempts, i, k,&
                     trimer, nonzero, choice
      logical(C_Bool) :: pop_loss(4)
      real(dp) :: dt, rho_q, rand, probs(rate_size)
      if (rho_q.gt.0.0) then
        n_attempts = n_sites
      else
        n_attempts = n_sites - 2
      end if
      do i = 1, n_attempts
        pop_loss = (/.false., .false., .false., .false./)
        trimer = randint(n_attempts)
        call update_rates(trimer, n_i(trimer), t)
        nonzero = 0
        ! need to check this is right lol
        probs = [(rates(k) * exp(-1.0_dp * rates(k) * dt), &
          k = (trimer - 1) * rate_size + 1, (trimer * rate_size) + 1)]
        do k = 1, size(probs)
          if (probs(k).gt.0.0_dp) then
            nonzero = nonzero + 1
          end if
        end do
        choice = randint(nonzero)
        nonzero = 0
        do k = 1, size(probs)
          if (probs(k).gt.0.0_dp) then
            nonzero = nonzero + 1
          end if
          if (nonzero.eq.choice) then
            choice = k
            exit
          end if
        end do
        call random_number(rand)
        if (rand.lt.probs(choice)) then
          call move(trimer, choice, pop_loss)
        end if
        if (any(pop_loss)) then
          do k = 1, size(pop_loss)
            if (pop_loss(k)) then
              ! append loss time and decay type to file here!
              write(20, '(E1.5, I1)') t, k
            end if
          end do
        end if
      end do
      t = t + dt
    end subroutine mc_step

    integer(ip) function kmc_step() bind(C) result(res)
      implicit none
      real(dp) :: r1, r2, k_tot
      integer(ip) :: k, l, i, process
      logical(C_Bool) :: pop_loss(4)
      if ((n_current.eq.0).and.(t.gt.2.0_dp * mu)) then
        ! do nothing
        return
        res = -1
      else
        pop_loss = .false.
        call random_number(r1)
        call random_number(r2)
        call bkl(r1, i, process, k_tot)
        call move(i, process, pop_loss)
        t = t - (1.0_dp / k_tot) * log(r2)
        if (any(pop_loss)) then
          do k = 1, size(pop_loss)
            if (pop_loss(k)) then
              write(20, '(E1.5, I1)') t, k
            end if
          end do
        end if
      end if
      res = 0
    end function kmc_step

end program iteration
