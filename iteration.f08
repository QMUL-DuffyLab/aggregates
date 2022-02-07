program iteration
  use iso_fortran_env
  implicit none
  integer, parameter :: dp = REAL64

  integer :: n_sites, rate_size
  real(dp) :: mu, fluence
  integer, dimension(:), allocatable :: n_i
  real(dp), dimension(:), allocatable :: rates, base_rates, pulse
  logical, dimension(:), allocatable :: quenchers

  !
  ! note - we can get around the loss times by simply printing them
  ! to a file as we go, i guess. but keeping track of excitons on the
  ! pre-quencher and quencher is more difficult
  !
  !

  contains

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
      real(dp) :: t, ft, xsec, sigma_ratio
      ! fortran slicing is inclusive on both ends and it's 1-based
      ! check this is correct!
      rates(ind:ind + rate_size) = base_rates(ind:ind + rate_size)
      if (t < 2.0_dp * mu) then
        t_index = int(t)
        ft = pulse(t_index)
        xsec = 1.1E-14
        sigma_ratio = 3.0_dp
        if ((1 + sigma_ratio) * n <= 24) then
          rates(ind + 1) = xsec * fluence * ft * ((24.0_dp - (1 + sigma_ratio) * n)/ 24.0_dp)
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
        rates(ind + rate_size) = rates(ind + rate_size) * (n * (n - 1) / 2.0_dp)
      end if
    end subroutine update_rates

    subroutine mc_step()
      implicit none
    end subroutine mc_step

end program iteration
