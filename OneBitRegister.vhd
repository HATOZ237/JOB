----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 09/18/2021 09:54:36 AM
-- Design Name: 
-- Module Name: BitRegister - Behavioral
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

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity OneBitRegister is
    Port ( RESET : in STD_LOGIC;
       CLK : in STD_LOGIC;
       D : in STD_LOGIC;
       EN : in STD_LOGIC;
       Q : out STD_LOGIC);
end OneBitRegister;

architecture Behavioral of OneBitRegister is

begin

Process(Reset, CLK)
Begin
    If (Reset='1') then
        Q<='0';
    Elsif (clk'event and clk='1') then
        If (EN='1') then
            Q<=D;
        End if;
    End if;
End process;

end Behavioral;