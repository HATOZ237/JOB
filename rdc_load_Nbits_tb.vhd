----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10/14/2021 11:06:30 AM
-- Design Name: 
-- Module Name: rdc_load_Nbits_tb - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library logic_com;
use logic_com.All;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity rdc_load_Nbits_tb is
end rdc_load_Nbits_tb;

architecture tb of rdc_load_Nbits_tb is

constant clk_period : time := 5 ns;
constant N : integer := 10;
signal RESET : STD_LOGIC;
signal CLK : STD_LOGIC := '0';
signal ENABLE : STD_LOGIC;
signal MODE : STD_LOGIC;
signal INPUT : STD_LOGIC;
signal LOAD : STD_LOGIC_VECTOR (N-1 downto 0);
signal OUTPUT : STD_LOGIC;

begin
rdc_load_Nbits : entity logic_com.rdc_load_Nbits
    generic map(N => 10)
    port map ( RESET => RESET,
               CLK => CLK,
               ENABLE => ENABLE,
               MODE => MODE,
               INPUT => INPUT,
               LOAD => LOAD,
               OUTPUT => OUTPUT);

    RESET <= '1', '0' after 5 ns;
    CLK <= not clk after clk_period;
    ENABLE <= '1';
    MODE <= '0', '1' after 400 ns, '0' after 600 ns;
    INPUT <= '1';
    LOAD <= "1010101010";

end tb;
