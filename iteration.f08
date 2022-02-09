program iteration
  use iso_fortran_env
  implicit none
  integer, parameter :: dp = REAL64

  integer :: n_sites, rate_size, n_current, n_pq, n_q
  real(dp) :: mu, fluence, t
  integer, dimension(:), allocatable :: n_i, pq, q
  integer, dimension(:, :), allocatable :: neighbours
  real(dp), dimension(:), allocatable :: rates, base_rates, pulse
  logical, dimension(:), allocatable :: quenchers

  ! type trimer
  !   integer

  !
  ! note - we can get around the loss times by simply printing them
  ! to a file as we go, i guess. but keeping track of excitons on the
  ! pre-quencher and quencher is more difficult
  !
  !

  contains

    function randint(upper) result(i)
      ! returns an integer between 1 and number for array indexing
      implicit none
      integer, intent(in) :: upper
      real :: r
      integer :: i
      call random_number(r)
      i = ceiling(upper * r)
    end function randint

    subroutine counts(array, c, n_nonzero)
      implicit none
      integer, dimension(:), intent(in) :: array
      integer, dimension(:) :: c
      integer :: i, j, n_nonzero
      c = 0
      n_nonzero = 0
      do i = 1, size(array)
        if (array(i).eq.0) then
          continue
        else
          do j = 1, i
            if (array(j).eq.array(i)) then
              c(i) = c(i) + 1
            end if
          end do
        end if
      end do
      do i = 1, size(array)
        if (c(i).gt.0) then
          n_nonzero = n_nonzero + 1
        end if
      end do
    end subroutine counts

    function bkl(rand) result(l)
      implicit none
      real(dp), intent(in) :: rand
      real(dp) :: k_tot
      real(dp), dimension(size(rates)) :: c_rates
      integer :: l, r, m, k
      ! update cumulative rates
      c_rates = [(sum(rates(1:k)), k = 1, size(rates))]
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
    end function bkl

    subroutine update_rates(ind, n, t)
      implicit none
      integer :: ind, n, t_index, k
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

    subroutine move(ind, process, pop_loss)
      implicit none
      integer, intent(in) :: ind, process
      integer :: nn, choice
      logical, intent(inout) :: pop_loss(4)
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
          pq(choice) = pq(n_pq)
          pq(n_pq) = -1
          n_pq = n_pq - 1
          call update_rates(ind, n_i(ind), t)
          call update_rates(nn, n_i(nn), t)
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
          call update_rates(ind, n_i(ind), t)
          pop_loss(1) = .true.
        else
          write(*, *) "Move function failed on pre-quencher."
        end if
      else if (ind.eq.(n_sites)) then
        ! quencher
      end if

    end subroutine move

    subroutine mc_step()
      implicit none
    end subroutine mc_step

end program iteration
