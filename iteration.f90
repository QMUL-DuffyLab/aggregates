program iteration
  use iso_fortran_env
  use iso_c_binding
  use mpi_f08
  implicit none
  integer, parameter :: dp = c_double
  integer, parameter :: short = c_short
  integer, parameter :: ip = c_int
  real, parameter :: pi = 3.1415926535

  character(len=200) :: params_file, rates_file, neighbours_file,&
    counts_file, prefix_long, emissive_str, seed_arg
  character(len=:), allocatable :: file_path, prefix
  logical(c_bool), dimension(:), allocatable :: is_quencher
  logical(c_bool) :: emissive(4)
  integer(ip) :: i, j, k, n_sites, max_neighbours, rate_size, n_current,&
    seed_size, max_count, curr_max_count, n_quenchers, mpierr, rank,&
    num_procs, seed_start
  integer(ip), dimension(1) :: maxloc_temp_index
  integer(ip), dimension(4) :: curr_counts
  integer(ip), dimension(4, 1) :: curr_locs
  integer(ip), dimension(:), allocatable :: n_i, n_pq, n_q, quenchers, seed
  integer(ip), dimension(:), allocatable :: neighbours_temp
  integer(ip), dimension(:, :), allocatable :: neighbours, counts
  real(dp) :: mu, n_per_t, t, dt, t_pulse, rho_q, xsec,&
    sigma, fwhm, start_time, end_time, binwidth, max_time
  real(dp), dimension(:), allocatable :: rates, base_rates, pulse,&
    bins
  real(dp), dimension(:, :), allocatable :: ds

  call MPI_Init(mpierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank,      mpierr)
  call MPI_Comm_size(MPI_COMM_WORLD, num_procs, mpierr)

  call init(rank)

  ! GCC docs say that random_number etc implement xoshiro1024*
  ! since F95, which should be more than enough for our purposes
  call random_seed(size=seed_size)
  allocate(seed(seed_size))

  ! rough estimate of how many counts we want per process
  ! to get good counts after reduction. this could be done
  ! exactly via MPI_REDUCE(MPI_SEND(maxval(whatever))) but
  ! we don't need exactly max_count counts, so who cares
  max_count = ceiling(float(max_count) / num_procs)
  if (rank.eq.0) then
    write(*, *) "max count per process = ", max_count
    call cpu_time(start_time)
  end if

  dt = 1.0_dp
  pulse = construct_pulse(mu, fwhm, dt, n_per_t, n_sites)
  ! base_rates(1) is where a generation rate will go
  ! base_rates(2) is stimulated emission
  ! base_rates(3) is the hopping rate
  ds = entropic_penalties(base_rates(3))

  seed = (seed_start * num_procs) + rank
  write(*, *) "Process ", rank, " has seed ", seed
  call random_seed(put=seed)
  i = 0
  curr_max_count = 0
  ! keep iterating till we get a decent number of counts
  do while (curr_max_count.lt.max_count)
    is_quencher = .false.
    call allocate_quenchers(n_quenchers, is_quencher, quenchers)
    call fix_base_rates()
    n_i = 0
    n_pq = 0
    n_q = 0
    n_current = 0
    t = 0.0_dp
    do while (t.lt.(size(pulse) * dt))
      ! ensure we generate excitons, otherwise
      ! n_current = 0 and we never start
      call mc_step(dt, n_quenchers)
    end do
    do while ((n_current.gt.0).and.(t.lt.max_time))
      call mc_step(dt, n_quenchers)
    end do
    i = i + 1

    if (mod(i, 100).eq.0) then
      ! every 100 iterations, check the current bin counts
      ! NB: to get exact numbers we'd have to add together all the emissive
      ! counts, and we could do that, but it's unecessary since we don't need
      ! exactly max_counts counts, it's just to get better statistics.
      ! hence, just take the current highest count for each emissive process.
      ! find the location (which bin) each is in. maxloc(curr_counts) gives us
      ! which of those is largest; assume this bin will give the largest sum.
      ! curr_locs(maxloc(curr_counts)) gives this bin. then 
      ! sum all the emissive decays in that bin
      curr_counts = 0
      curr_locs = 0
      do j = 1,4
        if (emissive(j)) then
          curr_counts(j) = maxval(counts(j, :))
          curr_locs(j, :) = int(maxloc(counts(j, :)))
        end if
      end do

      k = 0
      ! if all the counts are zero the indexing here will go funny,
      ! so check for that
      if (all(curr_counts.eq.0)) then
        curr_max_count = 0
      else
        do j = 1, 4
          if (emissive(j)) then
            ! maxloc returns an array - very irritating!!!!!
            maxloc_temp_index = curr_locs(maxloc(curr_counts), 1)
            k = k + counts(j, maxloc_temp_index(1))
          end if
        end do
        curr_max_count = k
      end if
      write(*, *) i, rank, [(maxval(counts(j, :)), j = 1, 4)],&
        "max_count = ", curr_max_count
    end if
    
  end do
  write(*, *) "Process ", rank, " has max count ",&
    curr_max_count

  ! write(*, *) "]."
  call MPI_Barrier(MPI_COMM_WORLD, mpierr)

  if (rank.eq.0) then
    call MPI_Reduce(MPI_IN_PLACE, counts, size(counts), MPI_INT,&
                      MPI_SUM, 0, MPI_COMM_WORLD, mpierr)
  else
    call MPI_Reduce(counts, counts, size(counts), MPI_INT,&
                      MPI_SUM, 0, MPI_COMM_WORLD, mpierr)
  end if

  if (rank.eq.0) then
    open(file=counts_file, unit=20)
    do j = 1, size(bins)
      write(20, '(F10.3, 4I10)') bins(j), [(counts(k, j), k = 1, 4)]
    end do
    close(20)

    call cpu_time(end_time)
    write(*, *) "Time elapsed: ", end_time - start_time
    write(*, *) "Number of iterations: ", i
  end if

  call deallocations()

  call MPI_Finalize(mpierr)

  contains

    subroutine init(rank)
      ! this is ugly lmao
      implicit none
      integer :: rank
      call get_command_argument(1, params_file)
      call get_command_argument(2, seed_arg)
      read(seed_arg, *) seed_start
      if (rank.eq.0) then
        write(*, *) "Seed start = ", seed_start
      end if
      open(file=params_file, unit=20)
      read(20, *) n_sites
      read(20, *) max_neighbours
      read(20, *) rho_q
      read(20, *) n_per_t
      read(20, *) mu
      read(20, *) fwhm
      read(20, *) binwidth
      read(20, *) max_count
      read(20, '(a)') emissive_str
      read(20, '(a)') rates_file
      read(20, '(a)') neighbours_file
      read(20, '(a)') prefix_long
      close(20)

      ! check which decays are emissive
      read(unit=emissive_str, fmt=*) emissive 
      ! note - rate_size is set to this because there are a maximum
      ! of six processes that can happen on any given site that aren't
      ! hopping (five on a pool chlorophyll, six on pq, four on q).
      ! hence, max_neighbours + 6 is always a long enough array to hold
      ! every possible rate on every possible site.
      ! i actually ignore the possibility of hopping on pq though, so in
      ! principle this could be reduced by one, but then we'd have to check
      ! that max_neighbours is > 0 and put the extra pq rate in the middle
      ! somewhere, and that seems pointless just to save a few bytes
      rate_size = max_neighbours + 6
      if (rank.eq.0) then
        write(*, '(a, I4)')     "n_sites    = ", n_sites
        write(*, '(a, I1)')     "max neigh  = ", max_neighbours
        write(*, '(a, F8.3)')   "rho_q      = ", rho_q
        write(*, '(a, ES10.3)') "n_per_t    = ", n_per_t
        write(*, '(a, F8.3)')   "mu         = ", mu
        write(*, '(a, F8.3)')   "fwhm       = ", fwhm
        write(*, '(a, F8.3)')   "binwidth   = ", binwidth
        write(*, '(a, I8)')     "max count  = ", max_count
        write(*, '(a, a)')      "rate file  = ", rates_file
        write(*, '(a, a)')      "neigh file = ", neighbours_file
      end if

      i = scan(rates_file, "/\", .true.)
      file_path = rates_file(:i)
      ! write(prefix_long, '(F4.2, a, ES8.2, a)') rho_q, "_", fluence, "_"
      prefix = trim(adjustl(prefix_long))
      if (rank.eq.0) then
        write(*, *) "File path = ", file_path
        write(*, *) "Prefix = ", prefix
      end if
      write(counts_file, '(a, a, a, a, a)') file_path,&
        prefix, trim(adjustl(seed_arg)), "_", "counts.dat"

      ! fortran's fine with 0-sized arrays so this is okay
      ! even in the detergent case where there are no neighbours
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

      open(file=rates_file, unit=20)
      read(20, *) base_rates
      close(20)

      ! one set of bins for each decay type; the bin array index will be
      ! something like floor(t / bin_size) + 1
      max_time = 10000.0_dp
      allocate(bins(int(max_time/binwidth)))
      allocate(counts(4, int(max_time/binwidth)))
      counts = 0
      do i = 1, size(bins)
        bins(i) = (i - 1) * binwidth
      end do

      allocate(is_quencher(n_sites))
      is_quencher = .false.
      n_quenchers = int(n_sites * rho_q)
      allocate(quenchers(n_quenchers))

    end subroutine init

    subroutine deallocations()
      implicit none
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
      deallocate(counts)
      deallocate(file_path)
      deallocate(prefix)
    end subroutine deallocations

    function entropic_penalties(hopping_rate) result (ds)
      implicit none
      integer :: i, j, nmax
      real(dp), dimension(:, :), allocatable :: ds
      real(dp) :: hopping_rate
      ! sum of boltzmann factors for average exciton energies
      ! over one monomer is 4.7; set this to 5, then multiply by 3 
      nmax = 15
      allocate(ds(0:nmax, 0:nmax))
      ds = 0.0
      do i = 0, nmax
        do j = 0, nmax
          ! entropy factor for n_donor = i, n_acceptor = j
          if (i.le.j) then
            ds(i, j) = hopping_rate * &
                     ((i * (15.0 - j)) / ((j + 1) * (15.0 - i + 1))) 
          else
            ds(i, j) = hopping_rate
          end if
          ! write(*, '(F10.6, a)', advance="no") ds(i, j), " "
        end do
        ! write(*, *)
      end do
    end function entropic_penalties

    function construct_pulse(mu, fwhm, dt, n_per_t,&
        n_sites) result(pulse)
      implicit none
      real(dp) :: mu, fwhm, dt, sigma, tmax, n_per_t
      real(dp), dimension(:), allocatable :: pulse
      integer :: i, n_sites
      tmax = 2.0_dp * mu
      allocate(pulse(int(tmax / dt)))
      sigma = fwhm / (2.0_dp * (sqrt(2.0_dp * log(2.0_dp))))
      do i = 1, int(tmax / dt)
        pulse(i) = (n_per_t) / (sigma * sqrt(2.0_dp * pi)) * &
          exp(-1.0_dp * ((i * dt) - mu)**2 / (sqrt(2.0_dp) * sigma)**2)
      end do
    end function construct_pulse

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
         ! if this trimer's already a quencher, reroll till it isn't
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
      integer(ip) :: ind, start, end_, n, t_index, k
      real(dp) :: t, ft, sigma_ratio, n_pigments, ann_fac
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
      end if
      rates = base_rates(start:end_)
      if (t < size(pulse) * dt) then
        ! no generation on a quencher
        if ((rate_type.eq."POOL").or.&
          ((rate_type.eq."PQ").and.(n.lt.1))) then
          ft = pulse(int(t / dt) + 1)
          ! ft = \sigma_eff * J(t) (done in pulse construction above)
          ! N' = 24.0_dp to start - could also be 42 (all chls) or
          ! 54 (all pigments including carotenoids)
          ! and we assume (at first) \sigma_SE = \sigma (thanks Einstein)
          ! absorption first: k_abs = (ft / N') * (N - n)
          rates(1) = (ft / 24.0_dp) * (24.0_dp - n)
          ! now stimulated emission: k_SE = \sigma_SE (ft / N') * n
          rates(2) = (ft / 24.0_dp) * n
        end if
      else
        rates(1) = 0.0_dp
        rates(2) = 0.0_dp
      end if
      ! max population on pq/q should be 1
      ! only applies if this site's a quencher anyway,
      ! otherwise this rate's already been zeroed
      if (is_quencher(ind)) then
        if (rate_type.ne."PQ".and.(n_pq(ind).eq.1)) then
            rates(rate_size - 2) = 0.0_dp
        else if (rate_type.eq."PQ".and.(n_q(ind).eq.1)) then
            rates(rate_size - 2) = 0.0_dp
        end if
      end if
      do k = 3, rate_size - 1
        rates(k) = rates(k) * n
        if (k.lt.(rate_size - 3)) then
          ! entropic factor due to populations
          ! on this trimer and its neighbours
          rates(k) = rates(k) * ds(n, n_i(neighbours(ind, k - 2)))
        end if
      end do
      ! they should be able to annihilate with
      ! other excitons in the corresponding pool
      if ((is_quencher(ind)).and.(rate_type.ne."Q")) then
        n = n_i(ind) + n_pq(ind)
      end if
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
        else if (process.eq.2) then
          ! stimulated emission
          n_i(ind)  = n_i(ind)  - 1
          n_current = n_current - 1
          ! pop_loss(2) = .true.
        else if ((process.gt.2).and.(process.lt.(rate_size - 3))) then
          ! hop to neighbour. -1 because of the generation rate
          nn = neighbours(ind, process - 2)
          n_i(ind) = n_i(ind) - 1
          n_i(nn)  = n_i(nn)  + 1
        else if (process.eq.rate_size - 3) then
          ! there will always be an empty rate due to
          ! there being an extra possible process on pq
          continue
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
          ! if this is a quencher, pool excitons can
          ! annihilate with pq excitons. this assumes
          ! the pre-quencher is always another chl!
          if (is_quencher(ind)) then
            choice = randint(n_pq(ind) + n_i(ind))
            if (choice.eq.n_pq(ind)) then
              n_pq(ind) = n_pq(ind) - 1
            else
              n_i(ind)  = n_i(ind)  - 1
            end if
          else
            n_i(ind)  = n_i(ind)  - 1
          end if
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
        else if (process.eq.2) then
          ! stimulated emission
          n_pq(ind) = n_pq(ind) - 1
          n_current = n_current - 1
          ! pop_loss(3) = .true.
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
          choice = randint(n_pq(ind) + n_i(ind))
          if (choice.eq.n_pq(ind)) then
            n_pq(ind) = n_pq(ind) - 1
          else
            n_i(ind)  = n_i(ind)  - 1
          end if
          n_current = n_current - 1
          pop_loss(1) = .true.
        else
          write(*, *) "Move function failed on pre-quencher."
        end if
      else if (trim(rate_type).eq."Q") then
        ! quencher - assume no absorption
        if (process.eq.(rate_size - 2)) then
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
      if ((n_pq(ind).gt.1).or.(n_q(ind).gt.1)) then
        write(*, *) "move set pq/q occupancy to > 1!!!", n_i(ind),&
        n_pq(ind), n_q(ind), ind, rate_type, rates, process
        stop
      end if
      if ((n_i(ind).lt.0).or.(n_pq(ind).lt.0).or.(n_q(ind).lt.0)) then
        write(*, *) "move set occupancy to < 0!!!", n_i(ind),&
        n_pq(ind), n_q(ind), ind, rate_type, rates, process
        stop
      end if
    end subroutine move

    subroutine mc_step(dt, n_quenchers)
      implicit none
      character(4) :: rate_type
      integer(ip) :: n_attempts, nonzero, choice,&
        i, j, k, n_quenchers, ri, trimer
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
          ! pick a possible move
          choice = randint(nonzero)
          ri = 0
          do k = 1, rate_size
            if (rates(k).gt.0.0_dp) then
              ri = ri + 1
            end if
            if (ri.eq.choice) then
              choice = k
              exit
            end if
          end do
          ! Poisson probability of one event occurring in this
          ! interval: let rates(k) -> \lambda and n = 1. Then
          ! p = exp(-n \lambda dt) \lambda^n / n!
          ! = \lambda exp(-\lambda dt), as below
          probs = [(rates(k) * exp(-1.0_dp * rates(k) * dt), &
            k = 1, rate_size)]
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
          ! increment the relevant bin for the histograms
          do k = 1, size(pop_loss)
            if (pop_loss(k)) then
              counts(k, floor(t / binwidth) + 1) = &
              counts(k, floor(t / binwidth) + 1) + 1
            end if
          end do
        end if
      end do
      t = t + dt
    end subroutine mc_step

end program iteration
